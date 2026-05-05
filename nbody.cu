#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// host arrays - the CPU side of things
vector3 *hVel, *hPos;
double  *mass;

// device arrays - the GPU side (declared extern in vector.h)
vector3 *d_hVel, *d_hPos;
double  *d_mass;

// just malloc the three host arrays
void initHostMemory(int numObjects)
{
    hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
    hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
    mass = (double  *)malloc(sizeof(double)  * numObjects);
}

void freeHostMemory()
{
    free(hVel);
    free(hPos);
    free(mass);
}

// allocate GPU memory and push the initial state up to the device
void initDeviceMemory(int numObjects)
{
    cudaMalloc((void**)&d_hPos, sizeof(vector3) * numObjects);
    cudaMalloc((void**)&d_hVel, sizeof(vector3) * numObjects);
    cudaMalloc((void**)&d_mass, sizeof(double)  * numObjects);

    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double)  * numObjects, cudaMemcpyHostToDevice);
}

void freeDeviceMemory()
{
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
}

// pull final positions and velocities back from the GPU so we can print them
void syncToHost(int numObjects)
{
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * numObjects, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * numObjects, cudaMemcpyDeviceToHost);
}

// fill the first NUMPLANETS+1 slots with solar system data
void planetFill()
{
    int i, j;
    double data[][7] = {SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
    for (i = 0; i <= NUMPLANETS; i++) {
        for (j = 0; j < 3; j++) {
            hPos[i][j] = data[i][j];
            hVel[i][j] = data[i][j+3];
        }
        mass[i] = data[i][6];
    }
}

// throw some random asteroids into the mix
void randomFill(int start, int count)
{
    int i, j;
    for (i = start; i < start + count; i++) {
        for (j = 0; j < 3; j++) {
            //hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
            //hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
            //prevents the number from going crazy high in the output
            hVel[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
            hPos[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
            mass[i]    = (double)rand() / RAND_MAX * MAX_MASS;
        }
    }
}

void printSystem(FILE *handle)
{
    int i, j;
    for (i = 0; i < NUMENTITIES; i++) {
        fprintf(handle, "pos=(");
        for (j = 0; j < 3; j++)
            fprintf(handle, "%lf,", hPos[i][j]);
        fprintf(handle, "),v=(");
        for (j = 0; j < 3; j++)
            fprintf(handle, "%lf,", hVel[i][j]);
        fprintf(handle, "),m=%lf\n", mass[i]);
    }
}

int main(int argc, char **argv)
{
    clock_t t0 = clock();
    int t_now;

    srand(1234); // fixed seed so results are reproducible
    initHostMemory(NUMENTITIES);
    planetFill();
    randomFill(NUMPLANETS + 1, NUMASTEROIDS);

#ifdef DEBUG
    printSystem(stdout);
#endif

    // ship the initial state to the GPU - it stays there the whole simulation
    initDeviceMemory(NUMENTITIES);

    for (t_now = 0; t_now < DURATION; t_now += INTERVAL) {
        compute();
    }

    // bring results back so we can print them
    syncToHost(NUMENTITIES);

    clock_t t1 = clock() - t0;

#ifdef DEBUG
    printSystem(stdout);
#endif
    printf("This took a total time of %f seconds\n", (double)t1 / CLOCKS_PER_SEC);

    freeDeviceMemory();
    freeHostMemory();
    return 0;
}