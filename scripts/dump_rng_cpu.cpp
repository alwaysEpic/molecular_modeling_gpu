// Standalone CPU RNG dump tool for validation.
// Generates N random numbers using the same Box-Muller transform as the simulation.
// Usage: ./dump_rng_cpu <count> > rng_cpu.csv
//
// Compile: g++ -O2 -o dump_rng_cpu scripts/dump_rng_cpu.cpp -lm

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Same Box-Muller as simulation_cpu.cpp
static float randn_h(float mu, float sigma) {
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;

  if (call == 1) {
    call = !call;
    return (mu + sigma * (float)X2);
  }

  do {
    U1 = -1 + ((float)rand() / RAND_MAX) * 2;
    U2 = -1 + ((float)rand() / RAND_MAX) * 2;
    W = U1 * U1 + U2 * U2;
  } while (W >= 1 || W == 0);

  mult = sqrt((-2 * log(W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (float)X1);
}

int main(int argc, char* argv[]) {
  int count = 10000;
  if (argc > 1) count = atoi(argv[1]);

  srand((unsigned)time(NULL));

  for (int i = 0; i < count; i++) {
    printf("%0.15f\n", randn_h(0, 1));
  }
  return 0;
}
