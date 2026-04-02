#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#include "common/params.h"
#include "simulation_gpu.h"

// Shared device function — used by both d_update and d_simulate_isolated
__device__ void d_reflection(float x1, float y1, float x0, float y0, float* x_new, float* y_new, float radius) {
  float x_diff = x1 - x0;
  float y_diff = y1 - y0;

  float a = x_diff * x_diff + y_diff * y_diff;
  float b = 2.0f * (x_diff * x0 + y_diff * y0);
  float c = x0 * x0 + y0 * y0 - radius * radius;
  float disc = b * b - 4.0f * a * c;
  float t_Param = (-b + sqrtf(disc)) / (2.0f * a);

  float x_t = x_diff * t_Param + x0;
  float y_t = y_diff * t_Param + y0;
  float phi = atan2f(y_t, x_t);
  float cos_phi = cosf(phi);
  float sin_phi = sinf(phi);

  float overshoot_x = x1 - x_t;
  float overshoot_y = y1 - y_t;
  float A_rho = cos_phi * overshoot_x + sin_phi * overshoot_y;
  float A_phi = -sin_phi * overshoot_x + cos_phi * overshoot_y;

  *x_new = x_t + cos_phi * (-A_rho) - sin_phi * A_phi;
  *y_new = y_t + sin_phi * (-A_rho) + cos_phi * A_phi;
}

// ---------------------------------------------------------------------------
// Per-step kernel — kept for future collision mode and everything/allhit paths.
// Positions live in global memory (d_x, d_y, d_z), accessible between steps.
// ---------------------------------------------------------------------------
__global__ void d_update(long long rng_seed, float diffStep, float driftStep, int iter, float* d_x, float* d_y,
                         float* d_z, int walls, float radius, int start_flag, int t_step, int* d_hit_flag,
                         int* d_hit_step, float* d_hit_x, float* d_hit_y, float* d_hit_z, float limit, float rec_rad,
                         float locx, float locy, float locz, int firsthit, float start_x, float start_y,
                         float start_z) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= iter) return;

  if (firsthit && d_hit_flag[idx]) return;

  curandStatePhilox4_32_10_t state;
  curand_init(rng_seed, idx, 0, &state);

  float4 rn = curand_normal4(&state);
  float x_next = diffStep * rn.x;
  float y_next = diffStep * rn.y;
  float z_next = diffStep * rn.z + driftStep;

  // Save pre-step positions for bridge correction and wall reflection
  float x_last = d_x[idx], y_last = d_y[idx], pz_old = d_z[idx];

  // Bridge uniform from 4th normal value
  float u_bridge = 0.5f * (1.0f + erff(rn.w * 0.70710678118f));
  float Ddt = diffStep * diffStep * 0.5f;

  if (start_flag == 0) {
    d_x[idx] = start_x;
    d_y[idx] = start_y;
    d_z[idx] = start_z;
  } else {
    d_x[idx] += x_next;
    d_y[idx] += y_next;
    d_z[idx] += z_next;
  }

  if (walls == 1) {
    if (sqrtf(d_x[idx] * d_x[idx] + d_y[idx] * d_y[idx]) > radius) {
      float refl_x, refl_y;
      d_reflection(x_last + x_next, y_last + y_next, x_last, y_last, &refl_x, &refl_y, radius);
      d_x[idx] = refl_x;
      d_y[idx] = refl_y;
    }
  }

  // Hit detection with Brownian bridge correction (matches long kernel)
  if (d_hit_flag[idx]) return;

  if (limit > 0) {
    if (d_z[idx] > limit) {
      d_hit_flag[idx] = 1;
      d_hit_step[idx] = t_step;
      d_hit_x[idx] = d_x[idx];
      d_hit_y[idx] = d_y[idx];
      d_hit_z[idx] = d_z[idx];
      return;
    } else if (start_flag && pz_old < limit && d_z[idx] < limit) {
      float p_cross = __expf(-(limit - pz_old) * (limit - d_z[idx]) / Ddt);
      if (u_bridge < p_cross) {
        d_hit_flag[idx] = 1;
        d_hit_step[idx] = t_step;
        d_hit_x[idx] = d_x[idx];
        d_hit_y[idx] = d_y[idx];
        d_hit_z[idx] = limit;
        return;
      }
    }
  } else {
    float dx = d_x[idx] - locx, dy = d_y[idx] - locy, dz = d_z[idx] - locz;
    float r_new = sqrtf(dx * dx + dy * dy + dz * dz);
    if (rec_rad > r_new) {
      d_hit_flag[idx] = 1;
      d_hit_step[idx] = t_step;
      d_hit_x[idx] = d_x[idx];
      d_hit_y[idx] = d_y[idx];
      d_hit_z[idx] = d_z[idx];
    } else if (start_flag) {
      float dx_old = x_last - locx, dy_old = y_last - locy, dz_old = pz_old - locz;
      float r_old = sqrtf(dx_old * dx_old + dy_old * dy_old + dz_old * dz_old);
      if (r_old > rec_rad) {
        float d_old = r_old - rec_rad;
        float d_new = r_new - rec_rad;
        float p_cross = __expf(-d_old * d_new / Ddt);
        if (u_bridge < p_cross) {
          d_hit_flag[idx] = 1;
          d_hit_step[idx] = t_step;
          d_hit_x[idx] = d_x[idx];
          d_hit_y[idx] = d_y[idx];
          d_hit_z[idx] = d_z[idx];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Long kernel — all timesteps in one launch, positions in registers.
// For isolated particles only (first-hit/limit mode, no everything/allhit).
// No global position arrays needed. RNG initialized once per particle.
// ---------------------------------------------------------------------------
__global__ void d_simulate_isolated(long long rng_seed, float diffStep, float driftStep, int iter, int numSteps,
                                    int walls, float radius, float limit, float rec_rad, float locx, float locy,
                                    float locz, int firsthit, float start_x, float start_y, float start_z,
                                    int* d_hit_flag, int* d_hit_step, float* d_hit_x, float* d_hit_y, float* d_hit_z) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= iter) return;

  // RNG: init once, state lives in registers across all timesteps
  curandStatePhilox4_32_10_t state;
  curand_init(rng_seed, idx, 0, &state);

  // Positions in registers — no global memory for particle positions
  float px = start_x, py = start_y, pz = start_z;
  int hit = 0;

  for (int t = 0; t < numSteps; t++) {
    if (firsthit && hit) break;

    float4 rn = curand_normal4(&state);
    float x_next = diffStep * rn.x;
    float y_next = diffStep * rn.y;
    float z_next = diffStep * rn.z + driftStep;

    // Save pre-step positions for bridge correction and wall reflection
    float x_last = px, y_last = py, pz_old = pz;
    px += x_next;
    py += y_next;
    pz += z_next;

    // Brownian bridge uniform: transform unused 4th normal to U(0,1)
    // via the normal CDF. erff() is a hardware instruction (MUFU) on CUDA.
    float u_bridge = 0.5f * (1.0f + erff(rn.w * 0.70710678118f));

    // D*dt for bridge probability formula (derived from diffStep = sqrt(2*D*dt))
    float Ddt = diffStep * diffStep * 0.5f;

    // Wall reflection
    if (walls && sqrtf(px * px + py * py) > radius) {
      float refl_x, refl_y;
      d_reflection(x_last + x_next, y_last + y_next, x_last, y_last, &refl_x, &refl_y, radius);
      px = refl_x;
      py = refl_y;
    }

    // Hit detection with Brownian bridge correction
    // The bridge detects crossings where both endpoints are on the same side
    // of the boundary but the continuous path crossed and returned.
    // P_cross = exp(-(c-a)*(c-b) / (D*dt))
    // Formula is drift-independent (Revuz & Yor, Prop. IX.3.8).
    // See docs/brownian_bridge.md for full derivation and references.
    if (!hit) {
      if (limit > 0) {
        if (pz > limit) {
          // Explicit crossing — endpoint past boundary
          d_hit_flag[idx] = 1;
          d_hit_step[idx] = t;
          d_hit_x[idx] = px;
          d_hit_y[idx] = py;
          d_hit_z[idx] = pz;
          hit = 1;
        } else if (pz_old < limit && pz < limit) {
          // Bridge correction: both endpoints below limit, but continuous
          // path may have crossed and returned (Gobet 2000, NAM 1984)
          float p_cross = __expf(-(limit - pz_old) * (limit - pz) / Ddt);
          if (u_bridge < p_cross) {
            d_hit_flag[idx] = 1;
            d_hit_step[idx] = t;
            d_hit_x[idx] = px;
            d_hit_y[idx] = py;
            d_hit_z[idx] = limit;  // particle crossed at the boundary
            hit = 1;
          }
        }
      } else {
        // Spherical receiver: tangent-plane approximation (NAM 1984)
        float dx = px - locx, dy = py - locy, dz = pz - locz;
        float r_new = sqrtf(dx * dx + dy * dy + dz * dz);
        if (rec_rad > r_new) {
          // Explicit crossing — endpoint inside sphere
          d_hit_flag[idx] = 1;
          d_hit_step[idx] = t;
          d_hit_x[idx] = px;
          d_hit_y[idx] = py;
          d_hit_z[idx] = pz;
          hit = 1;
        } else {
          // Bridge correction: both endpoints outside sphere
          float dx_old = x_last - locx, dy_old = y_last - locy, dz_old = pz_old - locz;
          float r_old = sqrtf(dx_old * dx_old + dy_old * dy_old + dz_old * dz_old);
          if (r_old > rec_rad) {
            float d_old = r_old - rec_rad;
            float d_new = r_new - rec_rad;
            float p_cross = __expf(-d_old * d_new / Ddt);
            if (u_bridge < p_cross) {
              d_hit_flag[idx] = 1;
              d_hit_step[idx] = t;
              d_hit_x[idx] = px;
              d_hit_y[idx] = py;
              d_hit_z[idx] = pz;
              hit = 1;
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host entry point — dispatches to long or wide (per-step) kernel
// ---------------------------------------------------------------------------
float run_gpu_simulation(const SimParams& p, FILE* fp_out) {
  int numSteps = lround(p.time_in / p.deltaT);
  int gridSize = (p.iter + p.blockSize - 1) / p.blockSize;

  // Precompute constants
  float diffStep = sqrtf(2.0f * p.Db * p.deltaT);
  float driftStep = p.velocity * p.deltaT;

  // Use long kernel for isolated first-hit/limit modes.
  // --wide flag forces the per-step (wide) kernel for interaction development.
  int use_long = !p.wide && (p.firsthit == 1 || p.limit > 0) && p.everything == 0 && p.allhit == 0;

  // Device hit detection arrays (needed by both paths)
  int *d_hit_flag, *d_hit_step;
  float *d_hit_x, *d_hit_y, *d_hit_z;
  cudaMalloc(&d_hit_flag, p.iter * sizeof(int));
  cudaMalloc(&d_hit_step, p.iter * sizeof(int));
  cudaMalloc(&d_hit_x, p.iter * sizeof(float));
  cudaMalloc(&d_hit_y, p.iter * sizeof(float));
  cudaMalloc(&d_hit_z, p.iter * sizeof(float));
  cudaMemset(d_hit_flag, 0, p.iter * sizeof(int));
  cudaMemset(d_hit_step, 0, p.iter * sizeof(int));

  // Timing
  cudaEvent_t start, stop;
  float pcost = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  if (use_long) {
    // ---------- Long kernel path ----------
    // Single launch, all timesteps inside kernel, positions in registers.
    // No d_x/d_y/d_z arrays needed.
    long long rng_seed = (p.seed != 0) ? p.seed : (long long)clock();

    d_simulate_isolated<<<gridSize, p.blockSize>>>(rng_seed, diffStep, driftStep, p.iter, numSteps, p.walls, p.radius,
                                                    p.limit, p.rec_rad, p.locx, p.locy, p.locz, p.firsthit, p.start_x,
                                                    p.start_y, p.start_z, d_hit_flag, d_hit_step, d_hit_x, d_hit_y,
                                                    d_hit_z);

    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess && p.verbose == 1)
      printf("Cuda error kernel -- %s\n", cudaGetErrorString(code));

  } else {
    // ---------- Wide (per-step) kernel path ----------
    // All particles advance together, positions in global memory.
    // Used for everything/allhit modes, --wide flag, and future interactions.
    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, p.iter * sizeof(float));
    cudaMalloc(&d_y, p.iter * sizeof(float));
    cudaMalloc(&d_z, p.iter * sizeof(float));

    // Pre-initialize positions on device (eliminates start_flag branch in kernel)
    {
      std::vector<float> init_x(p.iter, p.start_x);
      std::vector<float> init_y(p.iter, p.start_y);
      std::vector<float> init_z(p.iter, p.start_z);
      cudaMemcpy(d_x, init_x.data(), p.iter * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, init_y.data(), p.iter * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_z, init_z.data(), p.iter * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Host arrays only needed for everything/allhit per-step output
    int needs_per_step_output = p.everything == 1 || p.allhit == 1;
    float *x = nullptr, *y = nullptr, *z = nullptr;
    std::vector<int> last_GPU;
    if (needs_per_step_output) {
      x = (float*)malloc(p.iter * sizeof(float));
      y = (float*)malloc(p.iter * sizeof(float));
      z = (float*)malloc(p.iter * sizeof(float));
      last_GPU.resize(p.iter, 0);
    }

    for (int t_count = 0; t_count < numSteps; t_count++) {
      long long rng_seed = (p.seed != 0) ? p.seed + t_count : clock();

      d_update<<<gridSize, p.blockSize>>>(rng_seed, diffStep, driftStep, p.iter, d_x, d_y, d_z, p.walls, p.radius,
                                          1, t_count, d_hit_flag, d_hit_step, d_hit_x, d_hit_y, d_hit_z, p.limit,
                                          p.rec_rad, p.locx, p.locy, p.locz, p.firsthit, p.start_x, p.start_y,
                                          p.start_z);

      cudaError_t code = cudaGetLastError();
      if (code != cudaSuccess && p.verbose == 1)
        printf("Cuda error kernel -- %s\n", cudaGetErrorString(code));

      // Future: d_interactions<<<...>>>() would go here

      // Per-step output for everything/allhit modes only
      if (needs_per_step_output) {
        cudaMemcpy(x, d_x, p.iter * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(y, d_y, p.iter * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(z, d_z, p.iter * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i_count = 0; i_count < p.iter; i_count++) {
          if (p.allhit == 1) {
            if (p.limit > 0 && z[i_count] > p.limit) {
              fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count], t_count);
            }
            if ((p.limit == 0) && p.rec_rad > sqrtf((x[i_count] - p.locx) * (x[i_count] - p.locx) +
                                                     (y[i_count] - p.locy) * (y[i_count] - p.locy) +
                                                     (z[i_count] - p.locz) * (z[i_count] - p.locz))) {
              fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count], t_count);
            }
          } else if (p.everything == 1) {
            fprintf(fp_out, "%0.15f, %0.15f, %0.15f, ", x[i_count], y[i_count], z[i_count]);
          }
        }
        fprintf(fp_out, "\n");
      }
    }

    free(x);
    free(y);
    free(z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&pcost, start, stop);

  // Copy hit results and write CSV
  // Runs for: long kernel (always), wide kernel without per-step output (first-hit/limit with --wide)
  // Does NOT run for: wide kernel with per-step output (everything/allhit — CSV already written per step)
  int wrote_per_step = !use_long && (p.everything == 1 || p.allhit == 1);
  if (!wrote_per_step) {
    int* hit_flag = (int*)malloc(p.iter * sizeof(int));
    int* hit_step = (int*)malloc(p.iter * sizeof(int));
    float* hit_x = (float*)malloc(p.iter * sizeof(float));
    float* hit_y = (float*)malloc(p.iter * sizeof(float));
    float* hit_z = (float*)malloc(p.iter * sizeof(float));

    cudaMemcpy(hit_flag, d_hit_flag, p.iter * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hit_step, d_hit_step, p.iter * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hit_x, d_hit_x, p.iter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hit_y, d_hit_y, p.iter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hit_z, d_hit_z, p.iter * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < p.iter; i++) {
      if (hit_flag[i]) {
        fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, \n", hit_x[i], hit_y[i], hit_z[i], hit_step[i]);
      }
    }

    free(hit_flag);
    free(hit_step);
    free(hit_x);
    free(hit_y);
    free(hit_z);
  }

  cudaFree(d_hit_flag);
  cudaFree(d_hit_step);
  cudaFree(d_hit_x);
  cudaFree(d_hit_y);
  cudaFree(d_hit_z);

  return pcost * 0.001f;
}
