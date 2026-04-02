#ifndef CLI_H
#define CLI_H

#include <cmath>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include "params.h"

// Long-only option codes (no short flag)
enum LongOpt {
  OPT_START_X = 256,
  OPT_START_Y,
  OPT_START_Z,
  OPT_REC_X,
  OPT_REC_Y,
  OPT_REC_Z,
  OPT_VELOCITY,
  OPT_DB,
};

static const struct option cli_options[] = {
    // Output modes
    {"all-hit", no_argument, nullptr, 'a'},
    {"everything", no_argument, nullptr, 'e'},
    {"first-hit", no_argument, nullptr, 'f'},

    // Simulation parameters
    {"iterations", required_argument, nullptr, 'i'},
    {"time", required_argument, nullptr, 't'},
    {"deltat", required_argument, nullptr, 'd'},
    {"db", required_argument, nullptr, OPT_DB},
    {"velocity", required_argument, nullptr, OPT_VELOCITY},
    {"nodrift", no_argument, nullptr, 'n'},
    {"seed", required_argument, nullptr, 'S'},

    // Geometry
    {"walls", no_argument, nullptr, 'w'},
    {"radius", required_argument, nullptr, 'r'},
    {"limit", required_argument, nullptr, 'l'},
    {"start-x", required_argument, nullptr, OPT_START_X},
    {"start-y", required_argument, nullptr, OPT_START_Y},
    {"start-z", required_argument, nullptr, OPT_START_Z},
    {"rec-x", required_argument, nullptr, OPT_REC_X},
    {"rec-y", required_argument, nullptr, OPT_REC_Y},
    {"rec-z", required_argument, nullptr, OPT_REC_Z},

    // GPU / runtime
    {"blocksize", required_argument, nullptr, 'b'},
    {"cpu-compare", no_argument, nullptr, 'c'},
    {"wide", no_argument, nullptr, 'W'},

    // General
    {"verbose", no_argument, nullptr, 'v'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0},
};

static inline void displayUsage(const char* program_name) {
  printf("Molecular Communication Particle Simulator\n\n");
  printf("Usage: %s [OPTIONS]\n", program_name);

  printf("\nOutput Modes:\n");
  printf("  -f, --first-hit          Record first hit per particle (default if -f or -l given)\n");
  printf("  -a, --all-hit            Record all hits per particle\n");
  printf("  -e, --everything         Record all positions at every timestep (large files)\n");

  printf("\nSimulation Parameters:\n");
  printf("  -i, --iterations N       Number of particle paths [default: 1000]\n");
  printf("  -t, --time T             Simulation duration in seconds [default: 1E-3]\n");
  printf("  -d, --deltat DT          Time step in seconds [default: 1E-7]\n");
  printf("      --db D               Diffusion coefficient in m^2/s [default: 1E-11]\n");
  printf("      --velocity V         Drift velocity in m/s [default: 1E-4]\n");
  printf("  -n, --nodrift            Disable drift (sets velocity to 0)\n");
  printf("  -S, --seed N             RNG seed for reproducibility [default: 0 = time-based]\n");

  printf("\nGeometry:\n");
  printf("  -w, --walls              Enable cylindrical vessel walls\n");
  printf("  -r, --radius R           Vessel radius in meters [default: 8E-6]\n");
  printf("  -l, --limit D            1D planar limit distance in meters (z-axis)\n");
  printf("      --start-x X          Transmitter x position in meters [default: 0]\n");
  printf("      --start-y Y          Transmitter y position in meters [default: 0]\n");
  printf("      --start-z Z          Transmitter z position in meters [default: 0]\n");
  printf("      --rec-x X            Receiver x position in meters [default: 0]\n");
  printf("      --rec-y Y            Receiver y position in meters [default: 0]\n");
  printf("      --rec-z Z            Receiver z position in meters [default: 50E-9]\n");

  printf("\nGPU / Runtime:\n");
  printf("  -b, --blocksize N        CUDA block size [default: 256]\n");
  printf("  -c, --cpu-compare        Run CPU reference alongside GPU for comparison\n");
  printf("  -W, --wide               Force wide (per-step) kernel path\n");

  printf("\nGeneral:\n");
  printf("  -v, --verbose            Print detailed simulation parameters\n");
  printf("  -h, --help               Show this help message\n");

  exit(0);
}

static inline void inputError(const char* param) {
  printf("Error: %s must be positive\n", param);
  exit(1);
}

static inline SimParams parseArgs(int argc, char* argv[]) {
  SimParams p = defaultParams();
  int longindex = 0;
  int opt;

  while ((opt = getopt_long(argc, argv, "ab:cd:efhi:l:nr:S:t:vwW", cli_options, &longindex)) != -1) {
    switch (opt) {
      // Output modes
      case 'a': p.allhit = 1; break;
      case 'e': p.everything = 1; break;
      case 'f': p.firsthit = 1; break;

      // Simulation parameters
      case 'i':
        p.iter = atoi(optarg);
        if (p.iter <= 0) inputError("iterations");
        break;
      case 't':
        p.time_in = atof(optarg);
        if (p.time_in <= 0) inputError("time");
        break;
      case 'd':
        p.deltaT = atof(optarg);
        if (p.deltaT <= 0) inputError("deltat");
        break;
      case OPT_DB:
        p.Db = atof(optarg);
        if (p.Db <= 0) inputError("db");
        break;
      case OPT_VELOCITY:
        p.velocity = atof(optarg);
        break;
      case 'n':
        p.velocity = 0.0;
        break;
      case 'S':
        p.seed = atoll(optarg);
        break;

      // Geometry
      case 'w': p.walls = 1; break;
      case 'r':
        p.radius = atof(optarg);
        if (p.radius <= 0) inputError("radius");
        break;
      case 'l':
        p.limit = atof(optarg);
        if (p.limit < 0) inputError("limit");
        break;
      case OPT_START_X: p.start_x = atof(optarg); break;
      case OPT_START_Y: p.start_y = atof(optarg); break;
      case OPT_START_Z: p.start_z = atof(optarg); break;
      case OPT_REC_X: p.locx = atof(optarg); break;
      case OPT_REC_Y: p.locy = atof(optarg); break;
      case OPT_REC_Z: p.locz = atof(optarg); break;

      // GPU / runtime
      case 'b':
        p.blockSize = atoi(optarg);
        if (p.blockSize <= 0) inputError("blocksize");
        break;
      case 'c': p.compare = 1; break;
      case 'W': p.wide = 1; break;

      // General
      case 'v': p.verbose = 1; break;
      case 'h': displayUsage(argv[0]); break;
      default: displayUsage(argv[0]);
    }
  }

  // Validate positions are inside vessel when walls are enabled
  if (p.walls) {
    float start_r = sqrtf(p.start_x * p.start_x + p.start_y * p.start_y);
    if (start_r >= p.radius) {
      printf("Error: start position (%.2e, %.2e) is outside vessel radius %.2e\n",
             p.start_x, p.start_y, p.radius);
      exit(1);
    }
    float rec_r = sqrtf(p.locx * p.locx + p.locy * p.locy);
    if (rec_r >= p.radius) {
      printf("Error: receiver position (%.2e, %.2e) is outside vessel radius %.2e\n",
             p.locx, p.locy, p.radius);
      exit(1);
    }
  }

  return p;
}

static inline void printVerbose(const SimParams& p) {
  int numSteps = lround(p.time_in / p.deltaT);
  int gridSize = (p.iter + p.blockSize - 1) / p.blockSize;

  printf("Simulation Parameters:\n");
  printf("  Iterations:    %d\n", p.iter);
  printf("  Stop Time:     %0.10f s\n", p.time_in);
  printf("  Delta T:       %0.10f s\n", p.deltaT);
  printf("  Num Steps:     %d\n", numSteps);
  printf("  Db:            %0.15f m^2/s\n", p.Db);
  printf("  Velocity:      %0.10f m/s\n", p.velocity);
  printf("  RNG Seed:      %lld%s\n", p.seed, p.seed == 0 ? " (time-based)" : "");

  printf("Geometry:\n");
  printf("  Wall Radius:   %0.15f m\n", p.radius);
  printf("  Walls:         %s\n", p.walls ? "enabled" : "disabled");
  printf("  Limit:         %0.15f m\n", p.limit);
  if (p.start_x != 0 || p.start_y != 0 || p.start_z != 0) {
    printf("  Start:         (%0.6e, %0.6e, %0.6e)\n", p.start_x, p.start_y, p.start_z);
  }
  printf("  Receiver:      (%0.6e, %0.6e, %0.6e)\n", p.locx, p.locy, p.locz);

  printf("GPU:\n");
  printf("  Blocksize:     %d\n", p.blockSize);
  printf("  Gridsize:      %d\n", gridSize);
}

#endif
