#ifndef SIMULATION_GPU_H
#define SIMULATION_GPU_H

#include <stdio.h>

#include "common/params.h"

// Run GPU simulation. Returns elapsed time in seconds.
float run_gpu_simulation(const SimParams& p, FILE* fp_out);

#endif
