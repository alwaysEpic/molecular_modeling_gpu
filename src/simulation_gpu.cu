#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#include "common/params.h"
#include "simulation_gpu.h"

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

  // Reflect: negate radial component, keep tangential
  *x_new = x_t + cos_phi * (-A_rho) - sin_phi * A_phi;
  *y_new = y_t + sin_phi * (-A_rho) + cos_phi * A_phi;
}

__global__ void d_update(long long rng_seed, float diffStep, float driftStep, int iter, float* d_x, float* d_y,
                         float* d_z, int walls, float radius, int start_flag, int t_step, int* d_hit_flag,
                         int* d_hit_step, float* d_hit_x, float* d_hit_y, float* d_hit_z, float limit, float rec_rad,
                         float locx, float locy, float locz, int firsthit, float start_x, float start_y,
                         float start_z) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= iter) return;

  // Skip particles that already recorded a first-hit
  if (firsthit && d_hit_flag[idx]) return;

  // Per-call RNG: Philox on stack (registers), seeded by clock64() or fixed seed
  // This is 4-5x faster than persistent global memory state — see LEARNINGS.md
  curandStatePhilox4_32_10_t state;
  curand_init(rng_seed, idx, 0, &state);

  // Generate 4 normals at once (Philox natively produces 4x32-bit values,
  // curand_normal4 exploits this — ~30% faster than 3 separate calls)
  float4 rn = curand_normal4(&state);

  float x_next = diffStep * rn.x;
  float y_next = diffStep * rn.y;
  float z_next = diffStep * rn.z + driftStep;

  float x_last, y_last;
  if (walls == 1) {
    if (start_flag == 0) {
      x_last = start_x;
      y_last = start_y;
    } else {
      x_last = d_x[idx];
      y_last = d_y[idx];
    }
  }

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

      // refl_x/y are absolute world-space positions (built from wall intersection
      // point + reflected overshoot). Do NOT add x_last — it's already baked in.
      d_x[idx] = refl_x;
      d_y[idx] = refl_y;
    }
  }

  // Hit detection (device-side)
  if (d_hit_flag[idx]) return;

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

  // Precompute constants (same for every thread every timestep)
  float diffStep = sqrtf(2.0f * p.Db * p.deltaT);
  float driftStep = p.velocity * p.deltaT;

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
    // RNG seed: fixed seed or clock64() for each kernel call
    long long rng_seed = (p.seed != 0) ? p.seed + t_count : clock();

    d_update<<<gridSize, p.blockSize>>>(rng_seed, diffStep, driftStep, p.iter, d_x, d_y, d_z, p.walls, p.radius,
                                        start_flag, t_count, d_hit_flag, d_hit_step, d_hit_x, d_hit_y, d_hit_z,
                                        p.limit, p.rec_rad, p.locx, p.locy, p.locz, p.firsthit, p.start_x,
                                        p.start_y, p.start_z);

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
          if ((p.limit == 0) && p.rec_rad > sqrtf((x[i_count] - p.locx) * (x[i_count] - p.locx) +
                                                   (y[i_count] - p.locy) * (y[i_count] - p.locy) +
                                                   (z[i_count] - p.locz) * (z[i_count] - p.locz))) {
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
