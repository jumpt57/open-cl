#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "ocl_macros.h"
#include "cpu_bitmap.h"

#define DIM 1000

//Common defines 
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

typedef struct cuComplex {
	float   r;
	float   i;
} cuComplex;

cuComplex createComplex(float _r, float _i) {
	struct cuComplex tmp;
	tmp.r = _r;
	tmp.i = _i;

	return tmp;
}


float magnitude2(cuComplex z) {
	return z.r * z.r + z.i * z.i;
}

cuComplex multiply(cuComplex a, cuComplex b) {
	return createComplex(a.r * b.r - a.i * b.i, a.i * b.r + a.r * b.i);
}

cuComplex add(cuComplex a, cuComplex b) {
	return createComplex(a.r + b.r, a.i + b.i);
}

int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c = createComplex(-0.8, 0.156);
	cuComplex a = createComplex(jx, jy);

	int i = 0;
	for (i = 0; i<200; i++) {
		a = add(multiply(a, a), c);
		if (magnitude2(a) > 1000)
			return 0;
	}

	return 1;
}

void kernelcpu(unsigned char *ptr) {
	for (int y = 0; y <DIM; ++y) {
		for (int x = 0; x <DIM; ++x) {
			int offset = x + y * DIM;
			int juliaValue = julia(x, y);

			ptr[offset * 4 + 0] = juliaValue * 100;
			ptr[offset * 4 + 1] = juliaValue * 200;
			ptr[offset * 4 + 2] = juliaValue * 300;
			ptr[offset * 4 + 3] = 255;
		}
	}

}

int main(void) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *ptr = bitmap.get_ptr();

	kernelcpu(ptr);

	bitmap.display_and_exit();
	
	return 0;
}