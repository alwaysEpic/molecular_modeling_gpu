#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "common/params.h"
#include "simulation_gpu.h"

__device__ void d_reflection(float x1, float y1, float x0, float y0, double* x_new, double* y_new, float radius) {
  double x_diff = (x1 - x0);
  double y_diff = (y1 - y0);

  double a = x_diff * x_diff + y_diff * y_diff;
  double b = 2 * ((x_diff)*x0 + (y_diff)*y0);
  double c = x0 * x0 + y0 * y0 - (double)radius * radius;
  double t_Param = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);

  double x_t = (x_diff)*t_Param + x0;
  double y_t = (y_diff)*t_Param + y0;
  double phiAngle = atan2(y_t, x_t);

  double A_rho = cos(phiAngle) * (x1 - x_t) + sin(phiAngle) * (y1 - y_t);
  double A_phi = -sin(phiAngle) * (x1 - x_t) + cos(phiAngle) * (y1 - y_t);
  *x_new = x_t + (cos(phiAngle) * (-A_rho) - sin(phiAngle) * A_phi);
  *y_new = y_t + (sin(phiAngle) * (-A_rho) + cos(phiAngle) * A_phi);
}

__global__ void d_update(float Db, float deltaT, int iter, float* d_x, float* d_y, float* d_z, float velocity,
                         int walls, float radius, int start_flag, int t_step, int* d_hit_flag, int* d_hit_step,
                         float* d_hit_x, float* d_hit_y, float* d_hit_z, float limit, float rec_rad, float locx,
                         float locy, float locz, int firsthit) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= iter) return;

  // Skip particles that already recorded a first-hit
  if (firsthit && d_hit_flag[idx]) return;

  float x_next, y_next, z_next;
  float x_last, y_last;

  // Initialize RNG
  curandStatePhilox4_32_10_t state;
  curand_init(clock64(), idx, 0, &state);

  float r_x = curand_normal(&state);
  float r_y = curand_normal(&state);
  float r_z = curand_normal(&state);

  x_next = sqrt(2 * Db * deltaT) * r_x;
  y_next = sqrt(2 * Db * deltaT) * r_y;
  z_next = (sqrt(2 * Db * deltaT) * r_z) + velocity * deltaT;

  if (walls == 1) {
    if (start_flag == 0) {
      x_last = 0;
      y_last = 0;
    } else {
      x_last = d_x[idx];
      y_last = d_y[idx];
    }
  }

  if (start_flag == 0) {
    d_x[idx] = 0.0;
    d_y[idx] = 0.0;
    d_z[idx] = 0.0;
  } else {
    d_x[idx] += x_next;
    d_y[idx] += y_next;
    d_z[idx] += z_next;
  }

  if (walls == 1) {
    if (sqrtf(d_x[idx] * d_x[idx] + d_y[idx] * d_y[idx]) > radius) {
      double refl_x, refl_y;
      d_reflection(x_last + x_next, y_last + y_next, x_last, y_last, &refl_x, &refl_y, radius);

      d_x[idx] = x_last + refl_x;
      d_y[idx] = y_last + refl_y;
    }
  }

  // Hit detection (device-side)
  if (d_hit_flag[idx]) return;  // already hit (for non-firsthit modes that still track)

  // 1D Limit test
  if (limit > 0 && d_z[idx] > limit) {
    d_hit_flag[idx] = 1;
    d_hit_step[idx] = t_step;
    d_hit_x[idx] = d_x[idx];
    d_hit_y[idx] = d_y[idx];
    d_hit_z[idx] = d_z[idx];
    return;
  }

  // Spherical receiver test
  if (limit == 0) {
    float dx = d_x[idx] - locx;
    float dy = d_y[idx] - locy;
    float dz = d_z[idx] - locz;
    if (rec_rad > sqrtf(dx * dx + dy * dy + dz * dz)) {
      d_hit_flag[idx] = 1;
      d_hit_step[idx] = t_step;
      d_hit_x[idx] = d_x[idx];
      d_hit_y[idx] = d_y[idx];
      d_hit_z[idx] = d_z[idx];
    }
  }
}

float run_gpu_simulation(const SimParams& p, FILE* fp_out) {
  int numSteps = lround(p.time_in / p.deltaT);
  int gridSize = (p.iter + p.blockSize - 1) / p.blockSize;

  // Device position arrays
  float *d_x, *d_y, *d_z;
  cudaMalloc(&d_x, p.iter * sizeof(float));
  cudaMalloc(&d_y, p.iter * sizeof(float));
  cudaMalloc(&d_z, p.iter * sizeof(float));

  // Device hit detection arrays
  int *d_hit_flag, *d_hit_step;
  float *d_hit_x, *d_hit_y, *d_hit_z;
  cudaMalloc(&d_hit_flag, p.iter * sizeof(int));
  cudaMalloc(&d_hit_step, p.iter * sizeof(int));
  cudaMalloc(&d_hit_x, p.iter * sizeof(float));
  cudaMalloc(&d_hit_y, p.iter * sizeof(float));
  cudaMalloc(&d_hit_z, p.iter * sizeof(float));
  cudaMemset(d_hit_flag, 0, p.iter * sizeof(int));
  cudaMemset(d_hit_step, 0, p.iter * sizeof(int));

  // Use optimized path for first-hit and limit modes (no per-step memcpy)
  int use_optimized = (p.firsthit == 1 || p.limit > 0) && p.everything == 0 && p.allhit == 0;

  // Fallback: host arrays for per-step copy (everything/allhit modes only)
  float *x = nullptr, *y = nullptr, *z = nullptr;
  std::vector<int> last_GPU;
  if (!use_optimized) {
    x = (float*)malloc(p.iter * sizeof(float));
    y = (float*)malloc(p.iter * sizeof(float));
    z = (float*)malloc(p.iter * sizeof(float));
    last_GPU.resize(p.iter, 0);
  }

  int start_flag = 0;

  // Timing
  cudaEvent_t start, stop;
  float pcost = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int t_count = 0; t_count < numSteps; t_count++) {
    d_update<<<gridSize, p.blockSize>>>(p.Db, p.deltaT, p.iter, d_x, d_y, d_z, p.velocity, p.walls, p.radius,
                                        start_flag, t_count, d_hit_flag, d_hit_step, d_hit_x, d_hit_y, d_hit_z,
                                        p.limit, p.rec_rad, p.locx, p.locy, p.locz, p.firsthit);

    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess && p.verbose == 1)
      printf("Cuda error kernel -- %s\n", cudaGetErrorString(code));

    if (start_flag == 0) {
      start_flag = 1;
    }

    // Fallback path: per-step memcpy for everything/allhit modes
    if (!use_optimized) {
      cudaMemcpy(x, d_x, p.iter * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, p.iter * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z, p.iter * sizeof(float), cudaMemcpyDeviceToHost);

      for (int i_count = 0; i_count < p.iter; i_count++) {
        if ((p.firsthit == 1 && last_GPU[i_count] == 0) || p.allhit == 1) {
          if (p.limit > 0 && z[i_count] > p.limit) {
            fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count], t_count);
            last_GPU[i_count] = 1;
          }
          if ((p.limit == 0) && p.rec_rad > sqrt(pow(x[i_count] - p.locx, 2) + pow(y[i_count] - p.locy, 2) +
                                                  pow(z[i_count] - p.locz, 2))) {
            fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count], t_count);
            last_GPU[i_count] = 1;
          }
        } else if (p.everything == 1) {
          fprintf(fp_out, "%0.15f, %0.15f, %0.15f, ", x[i_count], y[i_count], z[i_count]);
        }
      }
      fprintf(fp_out, "\n");
    }
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&pcost, start, stop);

  // Optimized path: single copy of hit results, write CSV
  if (use_optimized) {
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

  // Cleanup
  if (!use_optimized) {
    free(x);
    free(y);
    free(z);
  }
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_hit_flag);
  cudaFree(d_hit_step);
  cudaFree(d_hit_x);
  cudaFree(d_hit_y);
  cudaFree(d_hit_z);

  return pcost * 0.001f;
}
