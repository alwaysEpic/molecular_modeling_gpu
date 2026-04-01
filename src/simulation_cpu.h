#ifndef SIMULATION_CPU_H
#define SIMULATION_CPU_H

#include <stdio.h>
#include "common/params.h"

// Run CPU reference simulation. Returns elapsed time in seconds.
double run_cpu_simulation(const SimParams& p, FILE* fp_out);

#endif
