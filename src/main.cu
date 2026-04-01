#include "common/cli.h"
#include "simulation_cpu.h"
#include "simulation_gpu.h"

int main(int argc, char* argv[]) {
  SimParams p = parseArgs(argc, argv);

  if (p.verbose == 1) {
    printVerbose(p);
  }

  FILE* fp_h = fopen("output_h.csv", "w+");
  FILE* fp_d = fopen("output_d_wide.csv", "w+");
  if (!fp_h || !fp_d) {
    printf("Error: could not open output files\n");
    exit(-1);
  }

  double scost = 0;
  if (p.compare == 1) {
    scost = run_cpu_simulation(p, fp_h);
  }

  float pcost = run_gpu_simulation(p, fp_d);

  if (p.verbose == 1) {
    printf("+++++++++++ GPU code execution time is %lfs\n", pcost);
  }
  if (p.compare == 1) {
    printf("+++++++++++ Serial code execution time (then/now) in seconds is %lf\n", scost);
    printf("+++++++++++ The speedup(SerialTimeCost / GPUTimeCost) is %lf\n", scost / pcost);
    printf("+++++++++++ The efficiency(Speedup / NumProcessorCores) is %lf\n", scost / pcost / 4);
  }

  fclose(fp_d);
  fclose(fp_h);

  return 0;
}
