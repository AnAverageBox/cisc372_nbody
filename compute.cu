#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "vector.h"
#include "config.h"

// device pointers live here, they're declared extern in vector.h
//vector3 *d_hPos, *d_hVel;
//double  *d_mass;

// ---------------------------------------------------------------------------
// Kernel 1: computeAccels
//   Each thread handles one (i,j) pair in the NxN acceleration matrix.
//   We use shared memory to cache the position and mass of the j-bodies
//   for each 16x16 tile so we're not hammering global memory constantly.
// ---------------------------------------------------------------------------
__global__ void computeAccels(vector3 *pos, double *mass,
                               vector3 *accels, int n)
{
    // pull in a tile of j-side positions and masses into shared mem
    __shared__ double s_pos[BLOCK_SIZE][3];
    __shared__ double s_mass[BLOCK_SIZE];

    int i = blockIdx.y * blockDim.y + threadIdx.y;  // which body is being affected
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // which body is doing the affecting

    // only the first row of threads in this block loads the shared data
    if (threadIdx.y == 0 && j < n) {
        s_pos[threadIdx.x][0] = pos[j][0];
        s_pos[threadIdx.x][1] = pos[j][1];
        s_pos[threadIdx.x][2] = pos[j][2];
        s_mass[threadIdx.x]   = mass[j];
    }
    __syncthreads(); // make sure everyone sees the shared data before we proceed

    if (i >= n || j >= n) return;

    // a body doesn't exert gravity on itself, just zero it out
    if (i == j) {
        accels[i * n + j][0] = 0.0;
        accels[i * n + j][1] = 0.0;
        accels[i * n + j][2] = 0.0;
    } else {
        // standard gravitational acceleration formula
        double dist[3];
        dist[0] = pos[i][0] - s_pos[threadIdx.x][0];
        dist[1] = pos[i][1] - s_pos[threadIdx.x][1];
        dist[2] = pos[i][2] - s_pos[threadIdx.x][2];

        double mag_sq = dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2];
        double mag    = sqrt(mag_sq);
        double amag   = -1.0 * GRAV_CONSTANT * s_mass[threadIdx.x] / mag_sq;

        accels[i * n + j][0] = amag * dist[0] / mag;
        accels[i * n + j][1] = amag * dist[1] / mag;
        accels[i * n + j][2] = amag * dist[2] / mag;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: updateBodies
//   One thread per body. Just walks across its row in the accels matrix,
//   sums everything up, then updates velocity and position.
// ---------------------------------------------------------------------------
__global__ void updateBodies(vector3 *pos, vector3 *vel,
                              vector3 *accels, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // sum up all the gravitational nudges on body i
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int j = 0; j < n; j++) {
        ax += accels[i * n + j][0];
        ay += accels[i * n + j][1];
        az += accels[i * n + j][2];
    }

    // update velocity then position using the total acceleration
    vel[i][0] += ax * INTERVAL;
    vel[i][1] += ay * INTERVAL;
    vel[i][2] += az * INTERVAL;

    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

// ---------------------------------------------------------------------------
// compute(): called once per timestep from main.
//   d_hPos/d_hVel/d_mass are already on the GPU from the setup in nbody.cu,
//   so we just need to allocate the accels scratch space, run the two
//   kernels, and clean up.
// ---------------------------------------------------------------------------
void compute()
{
    int n = NUMENTITIES;

    // temp storage for the NxN acceleration matrix on the GPU
    vector3 *d_accels;
    cudaMalloc((void**)&d_accels, sizeof(vector3) * n * n);

    // --- launch kernel 1: fill the acceleration matrix ---
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    computeAccels<<<grid, block>>>(d_hPos, d_mass, d_accels, n);
    cudaDeviceSynchronize();

    // --- launch kernel 2: sum rows and update positions/velocities ---
    int threads1d = 256;
    int blocks1d  = (n + threads1d - 1) / threads1d;

    updateBodies<<<blocks1d, threads1d>>>(d_hPos, d_hVel, d_accels, n);
    cudaDeviceSynchronize();

    // done with the accels scratch space
    cudaFree(d_accels);
}