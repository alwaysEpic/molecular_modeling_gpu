#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include "common/params.h"
#include "simulation_gpu.h"

__device__ void d_reflection(float x1, float y1, float x0, float y0, double * x_new, double * y_new, float velocity, float deltaT, float radius){

  double x_diff = (x1-x0);
  double y_diff = (y1-y0);

   double a = powf(x_diff,2) + powf(y_diff,2);
   double b = 2*((x_diff)*x0 + (y_diff)*y0);
   double c = powf(x0,2) + powf(y0,2) - powf(radius,2);
   double t_Param = (-b + sqrtf(powf(b,2)-4*a*c))/(2*a);//(2*c)/(-b+sqrtf(powf(b,2)-4*a*c));
   //(-b + sqrtf(powf(b,2)-4*a*c)/(2*a));//Quadradic equation

   //(-b + np.sqrt(b**2-4*a*c))/(2*a)

   double x_t = (x_diff) * t_Param + x0; //where collision with wall occurs
   double y_t = (y_diff) * t_Param + y0;
   double phiAngle = atan2f(y_t,x_t);

   double A_rho = cosf(phiAngle) * (x1-x_t) + sinf(phiAngle)*(y1-y_t);
   double A_phi = -sinf(phiAngle) * (x1-x_t) + cosf(phiAngle)*(y1-y_t);
   *x_new = x_t + (cosf(phiAngle)*(-A_rho) - sinf(phiAngle)*A_phi);
   *y_new = y_t + (sinf(phiAngle)*(-A_rho) + cosf(phiAngle)*A_phi);


   //float dot = (x * n_x) + (y * n_y) + (z * n_z);
   //printf("Dot: %0.15f\nx,y,z,Nx,Ny,Nz: %0.15f, %0.15f,%0.15f,%0.15f,%0.15f,%0.15f\n",dot,x,y,z,*n_x,*n_y,*n_z);
   //n_x = (x - 2) * dot * (n_x);
   //n_y = (y - 2) * dot * (n_y);
   //n_z = (z - 2) * dot * (n_z) + (velocity*deltaT);
   //printf("New(IN LOOP)XYZ: %0.15f, %0.15f,%0.15f\n",*n_x,*n_y,*n_z);
  //return vector - 2 * Vector3.Dot(vector, normal) * normal;
}

//Update function for GPU
__global__ void d_update(float Db, float deltaT, int iter, float * d_x, float * d_y, float * d_z, double * x_new, double * y_new, float velocity, int walls, float radius, int start_flag){//,long long thread){

  //1d Grid of 1D blocks
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x_next, y_next, z_next;
  //float x_norm, y_norm, z_norm;
  float x_last, y_last;//, z_last;

    if(idx < iter){ //was after _next before walls

    // Initialize RNG https://gist.github.com/akiross/17e722c5bea92bd2c310324eac643df6
    curandStatePhilox4_32_10_t state;
    curand_init(clock64(), idx, 0, &state);


    float r_x = curand_normal(&state);//BoxMuller();//randn_d(0, 1, idx, iter);//curand_normal(&rng1);//[blockIdx.x]);
    float r_y = curand_normal(&state);//BoxMuller();//randn_d(0, 1, idx, iter);//randn_d(0, 1, idx, iter);//curand_normal(&rng1);//[blockIdx.x]);
    float r_z = curand_normal(&state);//BoxMuller();//randn_d(0, 1, idx, iter);//randn_d(0, 1, idx, iter);//curand_normal(&rng1);//[blockIdx.x]);

    x_next = sqrt(2*Db*deltaT)*r_x;
    y_next = sqrt(2*Db*deltaT)*r_y;
    z_next = (sqrt(2*Db*deltaT)*r_z) + velocity*deltaT;//h_zDrift(velocity, deltaT);

    if(walls == 1){
      if(start_flag == 0){
        x_last = 0;
        y_last = 0;
      }
      else{
        x_last = d_x[idx];
        y_last = d_y[idx];
      }
    }

    if(start_flag == 0){
      d_x[idx] = 0.0;
      d_y[idx] = 0.0;
      d_z[idx] = 0.0;
    }
    else{
      d_x[idx] += x_next;
      d_y[idx] += y_next;
      d_z[idx] += z_next;
    }

    if(walls == 1){
      if(sqrt(powf(abs(d_x[idx]),2) + powf(abs(d_y[idx]),2)) > radius){
        d_reflection(x_last+x_next,y_last+y_next, x_last, y_last, x_new, y_new, velocity, deltaT, radius);

        d_x[idx] = x_last + *x_new;
        d_y[idx] = y_last + *y_new;
        printf("uhhhh\n");

       /* while(sqrt(powf(d_x[idx],2) + powf(d_y[idx],2)) > radius){
          x_last -= sqrt(2*Db*deltaT)*r_x;
          y_last -= sqrt(2*Db*deltaT)*r_y;
          d_x[idx] = x_last;
          d_y[idx] = y_last;

          if(sqrt(powf(d_x[idx],2) + powf(d_y[idx],2)) > radius){
            d_reflection(x_last+x_next,y_last+y_next, x_last, y_last, x_new, y_new, velocity, deltaT, radius);

            d_x[idx] = *x_new;
            d_y[idx] = *y_new;
          }
        }*/
      }
    }
  }
}

float run_gpu_simulation(const SimParams& p, FILE* fp_out){

  int numSteps = p.time_in/p.deltaT;
  // BUG: integer division truncates before ceil, can miss threads at the tail.
  // Correct form: (p.iter + p.blockSize - 1) / p.blockSize
  int gridSize = ceil(p.iter/p.blockSize);

  //arrays to transfer device data too
  float * x = (float *)malloc(p.iter*sizeof(float));
  float * y = (float *)malloc(p.iter*sizeof(float));
  float * z = (float *)malloc(p.iter*sizeof(float));
  float * hit_time = (float *)malloc(p.iter*sizeof(float));
  float *d_x, *d_y, *d_z, *d_hit_time;
  double *x_new, *y_new;
  cudaMalloc( &d_x, p.iter*sizeof(float));
  cudaMalloc( &d_y, p.iter*sizeof(float));
  cudaMalloc( &d_z, p.iter*sizeof(float));
  cudaMalloc( &x_new, sizeof(double));
  cudaMalloc( &y_new, sizeof(double));
  cudaMalloc( &d_hit_time, p.iter*sizeof(float));

  //d_iterations
  int start_flag = 0;
  std::vector<int> last_GPU(p.iter);

  //timing start
  cudaEvent_t start, stop;
  float pcost = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for(int t_count = 0; t_count < numSteps; t_count++){
    //Kernel Call
    d_update<<<gridSize,p.blockSize>>>(p.Db, p.deltaT, p.iter, d_x, d_y, d_z, x_new, y_new, p.velocity, p.walls, p.radius, start_flag);

    //Cuda Errors
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess && p.verbose == 1)
        printf ("Cuda error kernel -- %s\n", cudaGetErrorString(code));

    //Sets starting position
    if(start_flag ==0){start_flag = 1;}

    //Copy data from CUDA memory
    cudaMemcpy(x, d_x, p.iter*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, p.iter*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, d_z, p.iter*sizeof(float), cudaMemcpyDeviceToHost);

    //Logic performed on data
    for(int i_count = 0; i_count < p.iter; i_count++){
      if((p.firsthit == 1 && last_GPU[i_count] == 0) || p.allhit == 1){
        //Limit 1D test
        if(p.limit > 0 && z[i_count] > p.limit){
          fprintf(fp_out,"%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count] , t_count); //(int)iter - 1
          last_GPU[i_count] = 1;
        }
        //Receiver test
        if((p.limit == 0) && p.rec_rad > sqrt(pow(x[i_count]-p.locx,2)+pow(y[i_count]-p.locy,2)+pow(z[i_count]-p.locz,2))){ //&& sphere == 1
          fprintf(fp_out,"%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count] , t_count); //(int)iter - 1
          last_GPU[i_count] = 1;
        }
      }
      else if(p.everything == 1){//Print all
        fprintf(fp_out,"%0.15f, %0.15f, %0.15f, ", x[i_count], y[i_count], z[i_count]); //(int)iter - 1
      }
    }
    //space between steps
    fprintf(fp_out, "\n");
  }

  //stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&pcost, start, stop);

  free(x);
  free(y);
  free(z);
  free(hit_time);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(x_new);
  cudaFree(y_new);
  cudaFree(d_hit_time);

  return pcost * 0.001f;
}
