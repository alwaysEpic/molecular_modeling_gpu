#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "timing.h" 
#include <curand.h>
#include <curand_kernel.h>
#include <getopt.h>
#include <string>
#include <vector>
using namespace std;

static const struct option options[] = {
//Options for CLI
  { "all-hit", no_argument, NULL, 'a' },
  { "blocksize", required_argument, NULL, 'b' },
  { "cpu-compare", no_argument, NULL, 'c' },
  { "deltat", required_argument, NULL, 's' },
  { "everything", no_argument, NULL, 'e' }, 
  { "first-hit", no_argument, NULL, 'f' },
  { "help", no_argument, NULL, 'h' },
  { "iterations", required_argument, NULL, 'i' },
  { "limit", no_argument, NULL, 'l' },
  { "nodrift", no_argument, NULL, 'n' },
  { "radius", required_argument, NULL, 'r' },
  { "time", required_argument, NULL, 't' },
  { "verbose", no_argument, NULL, 'v' },
  { "walls", no_argument, NULL, 'w' },
  { 0, 0, 0, 0 }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Display title
static void displayTitle() {
  puts("GPU Particle Simulation");
}

// Display version
static void displayVersion() {
  puts("Version 0.1");
}

// Display usage
static void displayUsage() {
  displayTitle();
  displayVersion();
  printf("\nUsage: ./output [OPTION]...\n\n");

  printf("Default Values:\n\tIterations = 1\n\tTime = 1\n\tradius = 0.0000005\n\n");

  printf("Options:\n");
  printf("\t -a --all-hit\t\tRecords all movement in hit zone\n");
  printf("\t -b --blocksize\t\tSet block size for GPU. Default 128\n");
  printf("\t -c --compare\t\tRuns the CPU and GPU version of code for comparision\n");
  printf("\t -s --deltat\t\tSet time delta. Default 1E-7\n");
  printf("\t -e --everything\tRecords all data (this will result in massive files)\n");
  printf("\t -f --first-hit\t\tRecords first hit movement only\n");
  printf("\t -i --iterations\tSets number of particle paths to be run\n");
  printf("\t -l --limit\t\tRecords first hit based on distance limit\n");
  printf("\t -n --nodrift\t\tTurns off laminar flow\n");
  printf("\t -r --radius\t\tSets the radius for the blood vessel walls. Must be used with --walls to have effect\n");
  printf("\t -t --time\t\tSets simulation time\n");
  printf("\t -v --verbose\t\tProvides addition data about the simulation\n");
  printf("\t -w --walls\t\tSimulates particle motion in blood vessel of default radius\n");

  exit(1);  
}

// Input error
static void inputError(const char in[20]){
  printf("%s input cannot be negative or zero!\n", in);
    exit(-1);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//functions
__host__ float randn_h(float mu, float sigma);
__device__ float randn_d(float mu, float sigma, long long threadID, int timeStepNum);
__host__ void h_update(float Db, float deltaT, int numSteps, float velocity, float * h_x, float * h_y, float * h_z, float rand_x, float rand_y, float rand_z);
__global__ void d_update(float Db, float deltaT, int iter, float * d_x, float * d_y, float * d_z, double * x_new, double * y_new,float velocity, int walls, float radius, int start_flag);
__device__ void d_reflection(float x1, float y1, float x0, float y0, double * x_new, double * y_new, float velocity, float deltaT, float radius);

int main (int argc, char * argv[]) {    

  //Default Variables
  int longindex = 0, iter = 1000;
  int opt, verbose = 0, compare = 0, walls = 0, firsthit = 0, allhit = 0, everything = 0, blockSize = 128;//, sphere = 0;
  float time_in = 1E-3, radius = 8E-6, deltaT = 1E-7, limit = 0.0, velocity = 1E-4, Db = 1E-11;//3E-7;//1 * 0.0001; //velocity, 1E-3, 3E-4, rad 5e-7
  float rec_rad = 10E-9, rec_distance = 50E-9, locx = 0.0, locy = 0.0, locz = rec_distance;


  //Command-line Switch
  while ((opt = getopt_long(argc, argv, "ab:cs:efhi:l:nr:t:vw", options, &longindex)) != -1) {
    switch (opt) {
      case 'a': allhit = 1; break;
      case 'c': compare = 1; break;
      case 'e': everything = 1; break;
      case 'f': firsthit = 1; break;
      case 'h': displayUsage(); break;
      case 'n': velocity = 0.0; break;
      case 'v': verbose = 1; break;
      case 'w': walls = 1; break;
      case 'b': blockSize = atoi(optarg); 
          if(blockSize < 0){inputError("Blocksize");}
          break;
      case 's': deltaT = atof(optarg); 
          if(deltaT < 0){inputError("DeltaT");}
          break;
      case 'i': iter = atoi(optarg); 
          if(iter < 0){inputError("Iterations");}
          break;
      case 'l': limit = atof(optarg); 
          if(limit < 0){inputError("Limit");}
          break;
      case 'r': radius = atof(optarg); 
          if(radius < 0){inputError("Radius");}
          break;  
      case 't': time_in = atof(optarg); 
          if(time_in < 0){inputError("Time");}
          break;
      case '?': displayUsage();
      default: displayUsage();
    }
  }

  //Variable output
  int numSteps = time_in/deltaT;
  int gridSize = ceil(iter/blockSize);
  if(verbose == 1){
    printf("Blocksize: %d\nGridsize: %d\n", blockSize, gridSize);
    printf("Db: %0.15f\n", Db);
    printf("Delta T: %0.10f\n", deltaT);
    printf("Distance Limit: %0.15f\n", limit);
    printf("Iterations: %d\n", iter);
    printf("Number of Steps (time/deltaT): %d\n", numSteps);
    printf("Wall Radius: %0.15f\n", radius);
    printf("Stop Time: %0.10f\n", time_in);
    printf("Velocity: %0.10f\n", velocity); 
  }

////Host Variables
  float * h_x = (float *)malloc(sizeof(float));
  float * h_y = (float *)malloc(sizeof(float));
  float * h_z = (float *)malloc(sizeof(float));
  //time_t r_t;
  //srand((unsigned) time(&r_t));
  float rand_x = 0, rand_y = 0, rand_z = 0;

////GPU Variables
  //arrays to transfer device data too
  float * x = (float *)malloc(iter*sizeof(float));
  float * y = (float *)malloc(iter*sizeof(float));
  float * z = (float *)malloc(iter*sizeof(float));
  float * hit_time = (float *)malloc(iter*sizeof(float));
  float *d_x, *d_y, *d_z, *d_hit_time;
  double *x_new, *y_new;
  cudaMalloc( &d_x, iter*sizeof(float));
  cudaMalloc( &d_y, iter*sizeof(float));
  cudaMalloc( &d_z, iter*sizeof(float));   
  cudaMalloc( &x_new, sizeof(double));
  cudaMalloc( &y_new, sizeof(double));  
  cudaMalloc( &d_hit_time, iter*sizeof(float)); 

//Timing Variables
  double now, then , scost1 = 0;
  float pcost = 0;
  cudaEvent_t start, stop;
  int last_CPU = 0;

//Output files
FILE * fp_h = fopen("output_h.csv", "w+");
FILE * fp_d = fopen("output_d_wide.csv", "w+");

////Serial Execution 
  if(compare == 1){
    
    //timing start
    then = currentTime(); 
    //Path loop
    for(int i = 0; i < iter; i++){
      *h_x = 0; *h_y = 0; *h_z =0; //reset for next run
      //steps loop
      for(int jj = 0; jj < numSteps; jj++){
        rand_x = randn_h(0,1);
        rand_y = randn_h(0,1);
        rand_z = randn_h(0,1);

        h_update(Db, deltaT, numSteps, velocity, h_x, h_y, h_z, rand_x, rand_y, rand_z);
      	if((firsthit == 1 && last_CPU == 0) || allhit == 1){
          
          //Receiver test
          if((limit == 0) && rec_rad > sqrt((*h_x-locx)*(*h_x-locx)+(*h_y-locy)*(*h_y-locy)+(*h_z-locz)*(*h_z-locz))){// && sphere == 1
            fprintf(fp_h,"%0.15f, %0.15f, %0.15f, %d, %d, \n", *h_x, *h_y, *h_z , i , jj); //(int)iter - 1
            last_CPU = 1;
          }
          //1D Limit test
          if(limit > 0 && *h_z > limit){
            fprintf(fp_h,"%0.15f, %0.15f, %0.15f, %d, %d, \n", *h_x, *h_y, *h_z , i , jj); //(int)iter - 1
            last_CPU = 1;
          }
        }
        else if(everything){//print all
        	fprintf(fp_h,"%0.15f, %0.15f, %0.15f\n", *h_x, *h_y, *h_z);
          //fprintf(fp_h,"%0.15f, %0.15f, %0.15f\n", rand_x, rand_y, rand_z);
      	}
      }
      last_CPU = 0;
    }

    //Timing end
    now = currentTime();
    scost1 = now - then;
  }
////GPU Execution
   
   //d_iterations
   int start_flag = 0;
   std::vector<int> last_GPU(iter);
   
   //timing start
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);

   for(int t_count = 0; t_count < numSteps; t_count++){
      //Kernel Call
      d_update<<<gridSize,blockSize>>>(Db, deltaT, iter, d_x, d_y, d_z, x_new, y_new, velocity, walls, radius, start_flag);
      
      //Cuda Errors
      cudaError_t code = cudaGetLastError();
      if (code != cudaSuccess && verbose == 1) 
          printf ("Cuda error kernel -- %s\n", cudaGetErrorString(code)); 
      
      //Sets starting position
      if(start_flag ==0){start_flag = 1;}
      
      //Copy data from CUDA memory
      cudaMemcpy(x, d_x, iter*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, iter*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z, iter*sizeof(float), cudaMemcpyDeviceToHost); 
      
      //Logic performed on data 
      for(int i_count = 0; i_count < iter; i_count++){
        if((firsthit == 1 && last_GPU[i_count] == 0) || allhit == 1){
          //Limit 1D test
          if(limit > 0 && z[i_count] > limit){
            fprintf(fp_d,"%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count] , t_count); //(int)iter - 1
            last_GPU[i_count] = 1;
          }
          //Receiver test
          if((limit == 0) && rec_rad > sqrt(pow(x[i_count]-locx,2)+pow(y[i_count]-locy,2)+pow(z[i_count]-locz,2))){ //&& sphere == 1
            fprintf(fp_d,"%0.15f, %0.15f, %0.15f, %d, \n", x[i_count], y[i_count], z[i_count] , t_count); //(int)iter - 1
            last_GPU[i_count] = 1;
          }
        }
        else if(everything == 1){//Print all
          fprintf(fp_d,"%0.15f, %0.15f, %0.15f, ", x[i_count], y[i_count], z[i_count]); //(int)iter - 1
        }
      }
      //space between steps
      fprintf(fp_d, "\n");
   }

   //stop time
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&pcost, start, stop);

//Timing results 
   if(verbose == 1){
    printf("+++++++++++ GPU code execution time is %lfs\n", pcost*0.001);
   }
   if(compare == 1){
     printf("+++++++++++ Serial code execution time (then/now) in seconds is %lf\n", scost1);
     printf("+++++++++++ The speedup(SerialTimeCost / GPUTimeCost) is %lf\n", scost1 / (pcost*0.001)); 
     printf("+++++++++++ The efficiency(Speedup / NumProcessorCores) is %lf\n", scost1 / (pcost*0.001) / 4); 
   }

   /*int timing = 1;
   FILE * fp_t = fopen("timing.csv", "a");
   if(timing == 1){fprintf(fp_t,"%lf, \n", pcost*0.001);}
   fclose(fp_t);*/
   
////Freeing the clans, closing up shop
   free(x);
   free(y);
   free(z);
   free(h_x);
   free(h_y);
   free(h_z);
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_z);
   cudaFree(x_new);
   cudaFree(y_new);
   fclose(fp_d);
   fclose(fp_h);

   return 0;
}
  
//https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
//Box-Muller transform for Randn for CPU
__host__ float randn_h(float mu, float sigma){

  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;
 
  if (call == 1){
      call = !call;
      return (mu + sigma * (float) X2);
    }
 
  do{
      U1 = -1 + ((float) rand () / RAND_MAX) * 2;
      U2 = -1 + ((float) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (float) X1);
}

//Update function for CPU
__host__ void h_update(float Db, float deltaT, int numSteps, float velocity, float * h_x, float * h_y, float * h_z, float rand_x, float rand_y, float rand_z){

  *h_x += sqrt(2*Db*deltaT)*rand_x;
  *h_y += sqrt(2*Db*deltaT)*rand_y;
  *h_z += (sqrt(2*Db*deltaT)*rand_z) + velocity*deltaT;
  
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



