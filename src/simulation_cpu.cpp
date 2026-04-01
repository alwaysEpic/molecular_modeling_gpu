#include "simulation_cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common/params.h"
#include "timing.h"

// https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
// Box-Muller transform for Randn for CPU
static float randn_h(float mu, float sigma) {
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;

  if (call == 1) {
    call = !call;
    return (mu + sigma * (float)X2);
  }

  do {
    U1 = -1 + ((float)rand() / RAND_MAX) * 2;
    U2 = -1 + ((float)rand() / RAND_MAX) * 2;
    W = U1 * U1 + U2 * U2;
  } while (W >= 1 || W == 0);

  mult = sqrt((-2 * log(W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (float)X1);
}

// Update function for CPU
static void h_update(float diffStep, float driftStep, float* h_x, float* h_y, float* h_z, float rand_x, float rand_y,
                     float rand_z) {
  *h_x += diffStep * rand_x;
  *h_y += diffStep * rand_y;
  *h_z += diffStep * rand_z + driftStep;
}

double run_cpu_simulation(const SimParams& p, FILE* fp_out) {
  int numSteps = lround(p.time_in / p.deltaT);

  // Seed RNG
  if (p.seed != 0) {
    srand((unsigned)p.seed);
  } else {
    srand((unsigned)time(nullptr));
  }

  // Precompute constants (same for every particle every timestep)
  float diffStep = sqrtf(2.0f * p.Db * p.deltaT);
  float driftStep = p.velocity * p.deltaT;

  float h_x, h_y, h_z;
  float rand_x = 0, rand_y = 0, rand_z = 0;
  int last_CPU = 0;

  // timing start
  double then = currentTime();
  // Path loop
  for (int i = 0; i < p.iter; i++) {
    h_x = p.start_x;
    h_y = p.start_y;
    h_z = p.start_z;
    // D*dt for bridge probability formula
    float Ddt = diffStep * diffStep * 0.5f;

    // steps loop
    for (int jj = 0; jj < numSteps; jj++) {
      rand_x = randn_h(0, 1);
      rand_y = randn_h(0, 1);
      rand_z = randn_h(0, 1);

      float h_z_old = h_z;
      float h_x_old = h_x, h_y_old = h_y;
      h_update(diffStep, driftStep, &h_x, &h_y, &h_z, rand_x, rand_y, rand_z);

      // Brownian bridge uniform for boundary crossing correction
      float u_bridge = (float)rand() / (float)RAND_MAX;

      if ((p.firsthit == 1 && last_CPU == 0) || p.allhit == 1) {
        if (p.limit > 0) {
          // 1D Limit: explicit crossing
          if (h_z > p.limit) {
            fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, %d, \n", h_x, h_y, h_z, i, jj);
            last_CPU = 1;
          }
          // Bridge correction: both endpoints below limit
          else if (h_z_old < p.limit && h_z < p.limit) {
            float p_cross = expf(-(p.limit - h_z_old) * (p.limit - h_z) / Ddt);
            if (u_bridge < p_cross) {
              fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, %d, \n", h_x, h_y, p.limit, i, jj);
              last_CPU = 1;
            }
          }
        } else {
          // Spherical receiver: explicit crossing
          float r_new = sqrtf((h_x - p.locx) * (h_x - p.locx) + (h_y - p.locy) * (h_y - p.locy) +
                              (h_z - p.locz) * (h_z - p.locz));
          if (p.rec_rad > r_new) {
            fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, %d, \n", h_x, h_y, h_z, i, jj);
            last_CPU = 1;
          } else {
            // Bridge correction: both endpoints outside sphere
            float r_old = sqrtf((h_x_old - p.locx) * (h_x_old - p.locx) + (h_y_old - p.locy) * (h_y_old - p.locy) +
                                (h_z_old - p.locz) * (h_z_old - p.locz));
            if (r_old > p.rec_rad) {
              float d_old = r_old - p.rec_rad;
              float d_new = r_new - p.rec_rad;
              float p_cross = expf(-d_old * d_new / Ddt);
              if (u_bridge < p_cross) {
                fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, %d, \n", h_x, h_y, h_z, i, jj);
                last_CPU = 1;
              }
            }
          }
        }
      } else if (p.everything) {
        fprintf(fp_out, "%0.15f, %0.15f, %0.15f\n", h_x, h_y, h_z);
      }
    }
    last_CPU = 0;
  }

  // Timing end
  double now = currentTime();
  double elapsed = now - then;

  return elapsed;
}
