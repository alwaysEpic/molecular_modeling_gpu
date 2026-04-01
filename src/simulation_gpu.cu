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

  double x_t = (x_diff)*t_Param + x0;  // where collision with wall occurs
  double y_t = (y_diff)*t_Param + y0;
  double phiAngle = atan2(y_t, x_t);

  double A_rho = cos(phiAngle) * (x1 - x_t) + sin(phiAngle) * (y1 - y_t);
  double A_phi = -sin(phiAngle) * (x1 - x_t) + cos(phiAngle) * (y1 - y_t);
  *x_new = x_t + (cos(phiAngle) * (-A_rho) - sin(phiAngle) * A_phi);
  *y_new = y_t + (sin(phiAngle) * (-A_rho) + cos(phiAngle) * A_phi);
}

// Update function for GPU
__global__ void d_update(float Db, float deltaT, int iter, float* d_x, float* d_y, float* d_z, double* x_new,
                         double* y_new, float velocity, int walls, float radius,
                         int start_flag) {  //,long long thread){

  // 1d Grid of 1D blocks
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x_next, y_next, z_next;
  float x_last, y_last;

  if (idx < iter) {
    // Initialize RNG https://gist.github.com/akiross/17e722c5bea92bd2c310324eac643df6
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
        d_reflection(x_last + x_next, y_last + y_next, x_last, y_last, x_new, y_new, radius);

        d_x[idx] = x_last + *x_new;
        d_y[idx] = y_last + *y_new;
      }
    }
  }
}

float run_gpu_simulation(const SimParams& p, FILE* fp_out) {
  int numSteps = lround(p.time_in / p.deltaT);
  int gridSize = (p.iter + p.blockSize - 1) / p.blockSize;

  // arrays to transfer device data too
  float* x = (float*)malloc(p.iter * sizeof(float));
  float* y = (float*)malloc(p.iter * sizeof(float));
  float* z = (float*)malloc(p.iter * sizeof(float));
  float *d_x, *d_y, *d_z;
  double *x_new, *y_new;
  cudaMalloc(&d_x, p.iter * sizeof(float));
  cudaMalloc(&d_y, p.iter * sizeof(float));
  cudaMalloc(&d_z, p.iter * sizeof(float));
  cudaMalloc(&x_new, sizeof(double));
  cudaMalloc(&y_new, sizeof(double));

  // d_iterations
  int start_flag = 0;
  std::vector<int> last_GPU(p.iter);

  // timing start
  cudaEvent_t start, stop;
  float pcost = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int t_count = 0; t_count < numSteps; t_count++) {
    // Kernel Call
    d_update<<<gridSize, p.blockSize>>>(p.Db, p.deltaT, p.iter, d_x, d_y, d_z, x_new, y_new, p.velocity, p.walls,
                                        p.radius, start_flag);

    // Cuda Errors
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess && p.verbose == 1)
      printf("Cuda error kernel -- %s\n", cudaGetErrorString(code));

    // Sets starting position
    if (start_flag == 0) {
      start_flag = 1;
    }

    // Copy data from CUDA memory
    cudaMemcpy(x, d_x, p.iter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, p.iter * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, d_z, p.iter * sizeof(float), cudaMemcpyDeviceToHost);

    // Logic performed on data
    for (int i_count = 0; i_count < p.iter; i_count++) {
      if ((p.firsthit == 1 && last_GPU[i_count] == 0) || p.allhit == 1) {
        // Limit 1D test
        if (p.limit > 0 && z[i_count] > p.limit) {
          fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count], t_count);
          last_GPU[i_count] = 1;
        }
        // Receiver test
        if ((p.limit == 0) &&
            p.rec_rad > sqrt(pow(x[i_count] - p.locx, 2) + pow(y[i_count] - p.locy, 2) + pow(z[i_count] - p.locz, 2))) {
          fprintf(fp_out, "%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count], t_count);
          last_GPU[i_count] = 1;
        }
      } else if (p.everything == 1) {  // Print all
        fprintf(fp_out, "%0.15f, %0.15f, %0.15f, ", x[i_count], y[i_count], z[i_count]);
      }
    }
    // space between steps
    fprintf(fp_out, "\n");
  }

  // stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&pcost, start, stop);

  free(x);
  free(y);
  free(z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(x_new);
  cudaFree(y_new);

  return pcost * 0.001f;
}
