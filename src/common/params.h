#ifndef PARAMS_H
#define PARAMS_H

struct SimParams {
  // Physics
  float Db;        // Diffusion coefficient (m^2/s)
  float velocity;  // Laminar flow velocity (m/s)
  float deltaT;    // Time step (s)
  float time_in;   // Simulation duration (s)
  float radius;    // Blood vessel radius (m)

  // Receiver
  float rec_rad;       // Receiver radius (m)
  float rec_distance;  // Receiver distance from origin (m)
  float locx;          // Receiver x position
  float locy;          // Receiver y position
  float locz;          // Receiver z position

  // Simulation
  int iter;       // Number of particle paths
  int blockSize;  // GPU block size
  float limit;    // Distance limit for 1D hit test
  long long seed; // RNG seed (0 = use time-based seed)

  // Starting position
  float start_x;
  float start_y;
  float start_z;

  // Flags
  int verbose;
  int compare;
  int walls;
  int firsthit;
  int allhit;
  int everything;
  int wide;  // Force wide (per-step) kernel path
};

static inline SimParams defaultParams() {
  SimParams p;

  // Physics
  p.Db = 1E-11f;
  p.velocity = 1E-4f;
  p.deltaT = 1E-7f;
  p.time_in = 1E-3f;
  p.radius = 8E-6f;

  // Receiver
  p.rec_rad = 10E-9f;
  p.rec_distance = 50E-9f;
  p.locx = 0.0f;
  p.locy = 0.0f;
  p.locz = p.rec_distance;

  // Simulation
  p.iter = 1000;
  p.blockSize = 256;
  p.limit = 0.0f;
  p.seed = 0;
  p.start_x = 0.0f;
  p.start_y = 0.0f;
  p.start_z = 0.0f;

  // Flags
  p.verbose = 0;
  p.compare = 0;
  p.walls = 0;
  p.firsthit = 0;
  p.allhit = 0;
  p.everything = 0;
  p.wide = 0;

  return p;
}

#endif
