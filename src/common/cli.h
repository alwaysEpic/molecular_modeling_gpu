#ifndef CLI_H
#define CLI_H

#include <cmath>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include "params.h"

static const struct option cli_options[] = {{"all-hit", no_argument, nullptr, 'a'},
                                            {"blocksize", required_argument, nullptr, 'b'},
                                            {"cpu-compare", no_argument, nullptr, 'c'},
                                            {"deltat", required_argument, nullptr, 's'},
                                            {"everything", no_argument, nullptr, 'e'},
                                            {"first-hit", no_argument, nullptr, 'f'},
                                            {"help", no_argument, nullptr, 'h'},
                                            {"iterations", required_argument, nullptr, 'i'},
                                            {"limit", no_argument, nullptr, 'l'},
                                            {"nodrift", no_argument, nullptr, 'n'},
                                            {"radius", required_argument, nullptr, 'r'},
                                            {"rec-x", required_argument, nullptr, 4},
                                            {"rec-y", required_argument, nullptr, 5},
                                            {"rec-z", required_argument, nullptr, 6},
                                            {"seed", required_argument, nullptr, 'S'},
                                            {"start-x", required_argument, nullptr, 1},
                                            {"start-y", required_argument, nullptr, 2},
                                            {"start-z", required_argument, nullptr, 3},
                                            {"time", required_argument, nullptr, 't'},
                                            {"verbose", no_argument, nullptr, 'v'},
                                            {"walls", no_argument, nullptr, 'w'},
                                            {nullptr, 0, nullptr, 0}};

static inline void displayUsage(const char* program_name) {
  printf("Particle Simulation — Version 0.1\n");
  printf("\nUsage: %s [OPTION]...\n\n", program_name);

  printf("Default Values:\n\tIterations = 1000\n\tTime = 1E-3\n\tradius = 8E-6\n\n");

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
  printf("\t    --rec-x\t\tSet receiver x position (m). Default 0\n");
  printf("\t    --rec-y\t\tSet receiver y position (m). Default 0\n");
  printf("\t    --rec-z\t\tSet receiver z position (m). Default rec_distance (50nm)\n");
  printf("\t -S --seed\t\tSet RNG seed for reproducibility. Default 0 (time-based)\n");
  printf("\t    --start-x\t\tSet particle starting x position (m). Default 0\n");
  printf("\t    --start-y\t\tSet particle starting y position (m). Default 0\n");
  printf("\t    --start-z\t\tSet particle starting z position (m). Default 0\n");
  printf("\t -t --time\t\tSets simulation time\n");
  printf("\t -v --verbose\t\tProvides addition data about the simulation\n");
  printf("\t -w --walls\t\tSimulates particle motion in blood vessel of default radius\n");

  exit(1);
}

static inline void inputError(const char in[20]) {
  printf("%s input cannot be negative or zero!\n", in);
  exit(-1);
}

static inline SimParams parseArgs(int argc, char* argv[]) {
  SimParams p = defaultParams();
  int longindex = 0;
  int opt;

  while ((opt = getopt_long(argc, argv, "ab:cs:efhi:l:nr:S:t:vw", cli_options, &longindex)) != -1) {
    switch (opt) {
      case 'a':
        p.allhit = 1;
        break;
      case 'c':
        p.compare = 1;
        break;
      case 'e':
        p.everything = 1;
        break;
      case 'f':
        p.firsthit = 1;
        break;
      case 'h':
        displayUsage(argv[0]);
        break;
      case 'n':
        p.velocity = 0.0;
        break;
      case 'v':
        p.verbose = 1;
        break;
      case 'w':
        p.walls = 1;
        break;
      case 'b':
        p.blockSize = atoi(optarg);
        if (p.blockSize <= 0) {
          inputError("Blocksize");
        }
        break;
      case 's':
        p.deltaT = atof(optarg);
        if (p.deltaT <= 0) {
          inputError("DeltaT");
        }
        break;
      case 'i':
        p.iter = atoi(optarg);
        if (p.iter <= 0) {
          inputError("Iterations");
        }
        break;
      case 'l':
        p.limit = atof(optarg);
        if (p.limit < 0) {
          inputError("Limit");
        }
        break;
      case 'r':
        p.radius = atof(optarg);
        if (p.radius < 0) {
          inputError("Radius");
        }
        break;
      case 'S':
        p.seed = atoll(optarg);
        break;
      case 4:
        p.locx = atof(optarg);
        break;
      case 5:
        p.locy = atof(optarg);
        break;
      case 6:
        p.locz = atof(optarg);
        break;
      case 1:
        p.start_x = atof(optarg);
        break;
      case 2:
        p.start_y = atof(optarg);
        break;
      case 3:
        p.start_z = atof(optarg);
        break;
      case 't':
        p.time_in = atof(optarg);
        if (p.time_in <= 0) {
          inputError("Time");
        }
        break;
      default:
        displayUsage(argv[0]);
    }
  }

  // Validate positions are inside vessel when walls are enabled
  if (p.walls) {
    float start_r = sqrtf(p.start_x * p.start_x + p.start_y * p.start_y);
    if (start_r >= p.radius) {
      printf("Error: start position (%.2e, %.2e) is outside vessel radius %.2e\n",
             p.start_x, p.start_y, p.radius);
      exit(-1);
    }
    float rec_r = sqrtf(p.locx * p.locx + p.locy * p.locy);
    if (rec_r >= p.radius) {
      printf("Error: receiver position (%.2e, %.2e) is outside vessel radius %.2e\n",
             p.locx, p.locy, p.radius);
      exit(-1);
    }
  }

  return p;
}

static inline void printVerbose(const SimParams& p) {
  int numSteps = lround(p.time_in / p.deltaT);
  int gridSize = (p.iter + p.blockSize - 1) / p.blockSize;
  printf("Blocksize: %d\nGridsize: %d\n", p.blockSize, gridSize);
  printf("Db: %0.15f\n", p.Db);
  printf("Delta T: %0.10f\n", p.deltaT);
  printf("Distance Limit: %0.15f\n", p.limit);
  printf("Iterations: %d\n", p.iter);
  printf("Number of Steps (time/deltaT): %d\n", numSteps);
  printf("Wall Radius: %0.15f\n", p.radius);
  printf("Stop Time: %0.10f\n", p.time_in);
  printf("Velocity: %0.10f\n", p.velocity);
  printf("RNG Seed: %lld%s\n", p.seed, p.seed == 0 ? " (time-based)" : "");
  if (p.start_x != 0 || p.start_y != 0 || p.start_z != 0) {
    printf("Start Position: (%0.10f, %0.10f, %0.10f)\n", p.start_x, p.start_y, p.start_z);
  }
}

#endif
