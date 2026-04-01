#include "common/cli.h"
#include "simulation_cpu.h"

int main(int argc, char* argv[]) {
  SimParams p = parseArgs(argc, argv);

  if (p.verbose == 1) {
    printVerbose(p);
  }

  FILE* fp_h = fopen("output_h.csv", "w+");
  if (!fp_h) {
    printf("Error: could not open output file\n");
    exit(-1);
  }

  double scost = run_cpu_simulation(p, fp_h);

  if (p.verbose == 1) {
    printf("+++++++++++ CPU code execution time is %lfs\n", scost);
  }

  fclose(fp_h);

  return 0;
}
