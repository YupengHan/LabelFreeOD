#include <stdio.h>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

void testmain(int size, int *c) 
{
	int *a, *b; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c;
	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); a[0]=1;
	b = (int *)malloc(size); b[0]=4;
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	// Launch add() kernel on GPU
	add<<<1,1>>>(d_a, d_b, d_c);
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	// Cleanup
	free(a); free(b);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return;
}


__global__ void para_find_loc_fine (float* pts,
                            int ptnum,
                            int* scores,
                            float* xyz_limits) {
    int d_ix = blockIdx.x * blockDim.x + threadIdx.x;
    int d_iy = blockIdx.y * blockDim.y + threadIdx.y;
    int d_iz = blockIdx.z * blockDim.z + threadIdx.z;
    //printf("d_ix: %d d_iy: %d d_iz: %d\n", d_ix, d_iy, d_iz);
    float start_x = xyz_limits[0];
    float start_y = xyz_limits[2];
    float start_z = xyz_limits[4];


    //printf("start_x: %.0f start_y: %.0f start_z: %.0f\n", start_x, start_y, start_z);

    float end_x = xyz_limits[1];
    float end_y = xyz_limits[3];
    float end_z = xyz_limits[5];

    
    float cx = start_x + d_ix*10;
    float cy = start_y + d_iy*10;
    float cz = start_z + d_iz*10;

    if (cx > end_x || cy > end_y || cz > end_z) {
        //printf("cx: %.0f cy: %.0f cz: %.0f end_x: %.0f end_y: %.0f end_z: %.0f ", cx, cy, cz, end_x, end_y, end_z);
        scores[d_ix*100*400+d_iy*400+d_iz] = 0;
        return;
    }
    //printf("cx: %.0f cy: %.0f cz: %.0f end_x: %.0f end_y: %.0f end_z: %.0f \n", cx, cy, cz, end_x, end_y, end_z);
    
    
    
    int cnt = 0;
    for(int i = 0; i < ptnum; i++) {
        float tx = pts[i*3];
        float ty = pts[i*3+1];
        float tz = pts[i*3+2];
        if (tz > cz) continue;
        float d2c = sqrt((tx-cx)*(tx-cx) + (ty-cy)*(ty-cy) + (tz-cz)*(tz-cz));
        //printf("tx: %.0f ty: %.0f tz: %.0f          d2c: %.0f\n", tx, ty, tz, d2c);  
        
        /*
        if (d2c < 1000) {
            printf("tx: %.0f ty: %.0f tz: %.0f          d2c: %.0f\n", tx, ty, tz, d2c);  
        }
        */
        //printf("tx: %.0f ty: %.0f tz: %.0f cx: %.0f cy: %.0f cz: %.0f \n", tx, ty, tz, cx, cy, cz);
        //printf("tx: %.0f ty: %.0f tz: %.0f          d2c: %.0f\n", tx, ty, tz, d2c);
        
        
        
        /*
		if (d2c >= 50 && d2c <= 54 ) {
            cnt += 1;
        }
		*/
		
		if (d2c >= 34 && d2c <= 37 ) {
            cnt += 1;
        }
		
		
    }
    scores[d_ix*100*400+d_iy*400+d_iz] = cnt;
}


__global__ void find_best_score (int* scores,
                                float* xyz_limits,
                                float* device_pred_xyz) {
    int c_best = 0;
    device_pred_xyz[0] = -10000;
    device_pred_xyz[1] = -10000;
    device_pred_xyz[2] = -10000;


    int ixmax = int((xyz_limits[1] - xyz_limits[0])/10);
    if (ixmax > 100) ixmax = 100;
    int iymax = int((xyz_limits[3] - xyz_limits[2])/10);
    if (iymax > 100) iymax = 100;
    int izmax = int((xyz_limits[5] - xyz_limits[4])/10);
    if (izmax > 400) izmax = 400;
    //printf("ixmax : %d;  iymax : %d;  izmax : %d\n", ixmax, iymax, izmax);

    for (int ix = 0; ix < ixmax; ix++) {
        for (int iy = 0; iy < iymax; iy++) {
            for (int iz = 0; iz < izmax; iz++) {
                //c_best = c_best > scores[ix*100*400+iy*400+iz] ? c_best : scores[ix*100*400+iy*400+iz];
                if (c_best < scores[ix*100*400+iy*400+iz]) {
                    c_best = scores[ix*100*400+iy*400+iz];
                    device_pred_xyz[0] = xyz_limits[0] + 10*ix;
                    device_pred_xyz[1] = xyz_limits[2] + 10*iy;
                    device_pred_xyz[2] = xyz_limits[4] + 10*iz;
                    //printf("Score: %d    x: %.0f    y: %.0f      z:%.0f \n", c_best, device_pred_xyz[0], device_pred_xyz[1], device_pred_xyz[2]);
                }
                
            }
        }
    }
}

void find_loc_fine(float* pts, int ptnum, int* scores, float* xyz_limits, float* device_pred_xyz) {

    
    //dim3 grid(10, 100, 1);
    //dim3 block(10, 1, 400);
    
    dim3 grid(100, 10, 8);
    dim3 block(1, 10, 50);
    para_find_loc_fine<<<grid, block>>>(pts, ptnum, scores, xyz_limits);


    find_best_score<<<1, 1>>>(scores, xyz_limits, device_pred_xyz);





    

    cudaDeviceSynchronize();
}


