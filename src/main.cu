#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "common/params.h"
#include "simulation_cpu.h"
#include "simulation_gpu.h"

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

int main (int argc, char * argv[]) {

  //Default Variables
  SimParams p = defaultParams();
  int longindex = 0;
  int opt;

  //Command-line Switch
  while ((opt = getopt_long(argc, argv, "ab:cs:efhi:l:nr:t:vw", options, &longindex)) != -1) {
    switch (opt) {
      case 'a': p.allhit = 1; break;
      case 'c': p.compare = 1; break;
      case 'e': p.everything = 1; break;
      case 'f': p.firsthit = 1; break;
      case 'h': displayUsage(); break;
      case 'n': p.velocity = 0.0; break;
      case 'v': p.verbose = 1; break;
      case 'w': p.walls = 1; break;
      case 'b': p.blockSize = atoi(optarg);
          if(p.blockSize < 0){inputError("Blocksize");}
          break;
      case 's': p.deltaT = atof(optarg);
          if(p.deltaT < 0){inputError("DeltaT");}
          break;
      case 'i': p.iter = atoi(optarg);
          if(p.iter < 0){inputError("Iterations");}
          break;
      case 'l': p.limit = atof(optarg);
          if(p.limit < 0){inputError("Limit");}
          break;
      case 'r': p.radius = atof(optarg);
          if(p.radius < 0){inputError("Radius");}
          break;
      case 't': p.time_in = atof(optarg);
          if(p.time_in < 0){inputError("Time");}
          break;
      case '?': displayUsage();
      default: displayUsage();
    }
  }

  //Variable output
  int numSteps = p.time_in/p.deltaT;
  // BUG: integer division truncates before ceil, can miss threads at the tail.
  // Correct form: (p.iter + p.blockSize - 1) / p.blockSize
  int gridSize = ceil(p.iter/p.blockSize);
  if(p.verbose == 1){
    printf("Blocksize: %d\nGridsize: %d\n", p.blockSize, gridSize);
    printf("Db: %0.15f\n", p.Db);
    printf("Delta T: %0.10f\n", p.deltaT);
    printf("Distance Limit: %0.15f\n", p.limit);
    printf("Iterations: %d\n", p.iter);
    printf("Number of Steps (time/deltaT): %d\n", numSteps);
    printf("Wall Radius: %0.15f\n", p.radius);
    printf("Stop Time: %0.10f\n", p.time_in);
    printf("Velocity: %0.10f\n", p.velocity);
  }

//Output files
  FILE * fp_h = fopen("output_h.csv", "w+");
  FILE * fp_d = fopen("output_d_wide.csv", "w+");

////Serial Execution
  double scost = 0;
  if(p.compare == 1){
    scost = run_cpu_simulation(p, fp_h);
  }

////GPU Execution
  float pcost = run_gpu_simulation(p, fp_d);

//Timing results
  if(p.verbose == 1){
    printf("+++++++++++ GPU code execution time is %lfs\n", pcost);
  }
  if(p.compare == 1){
    printf("+++++++++++ Serial code execution time (then/now) in seconds is %lf\n", scost);
    printf("+++++++++++ The speedup(SerialTimeCost / GPUTimeCost) is %lf\n", scost / pcost);
    printf("+++++++++++ The efficiency(Speedup / NumProcessorCores) is %lf\n", scost / pcost / 4);
  }

  /*int timing = 1;
  FILE * fp_t = fopen("timing.csv", "a");
  if(timing == 1){fprintf(fp_t,"%lf, \n", pcost);}
  fclose(fp_t);*/

  fclose(fp_d);
  fclose(fp_h);

  return 0;
}
