#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include "ocl_macros.h"

//Common defines 
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

const char *sum_kernel =
"__kernel                                 \n"
"void sum_kernel(  __global int *A,       \n"
"                  __global int *B,       \n"
"                  __global int *C)       \n"
"{                                        \n"
"    *A = *B + *C;	     			  \n"
"}                                        \n";

int main(void) {

	cl_int clStatus; //Keeps track of the error values returned. 

	// Get platform and device information
	cl_platform_id * platforms = NULL;

	// Set up the Platform. Take a look at the MACROs used in this file. 
	// These are defined in common/ocl_macros.h
	OCL_CREATE_PLATFORMS(platforms);

	// Get the devices list and choose the type of device you want to run on
	cl_device_id *device_list = NULL;
	OCL_CREATE_DEVICE(platforms[0], DEVICE_TYPE, device_list);

	// Create OpenCL context for devices in device_list
	cl_context context;
	cl_context_properties props[3] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platforms[0],
		0
	};
	// An OpenCL context can be associated to multiple devices, either CPU or GPU
	// based on the value of DEVICE_TYPE defined above.
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateContext Failed...");

	// Create a command queue for the first device in device_list
	cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed...");

	// Allocate space for vectors A, B and C    
	int *A = (int *)malloc(sizeof(int));
	int *B = (int *)malloc(sizeof(int));
	int *C = (int *)malloc(sizeof(int));

	*A = 0;
	*B = 234;
	*C = 234;

	// Create memory buffers on the device for each vector
	cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &clStatus);
	cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &clStatus);
	cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &clStatus);

	// Copy the Buffer A and B to the device. We do a blocking write to the device buffer.
	clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, sizeof(int), A, 0, NULL, NULL);
	LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed...");
	clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, sizeof(int), B, 0, NULL, NULL);
	LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed...");

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&sum_kernel, NULL, &clStatus);
	LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed...");

	// Build the program
	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	if (clStatus != CL_SUCCESS)
		LOG_OCL_COMPILER_ERROR(program, device_list[0]);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "sum_kernel", &clStatus);

	// Set the arguments of the kernel. Take a look at the kernel definition in sum_event 
	// variable. First parameter is a constant and the other three are buffers.
	clStatus |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&A_clmem);
	clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&B_clmem);
	clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&C_clmem);
	LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed...");

	// Execute the OpenCL kernel on the list
	size_t global_size = 1;
	size_t local_size = 1;
	cl_event sum_event;
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &sum_event);
	LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed...");

	// Read the memory buffer C_clmem on the device to the host allocated buffer C
	// This task is invoked only after the completion of the event sum_event
	clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, sizeof(int), C, 1, &sum_event, NULL);
	LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed...");

	// Clean up and wait for all the comands to complete.
	clStatus = clFinish(command_queue);

	// Display the result to the screen
	printf("%i + %i = %i\n", *A, *B, *C);

	// Finally release all OpenCL objects and release the host buffers.
	clStatus = clReleaseKernel(kernel);
	clStatus = clReleaseProgram(program);
	clStatus = clReleaseMemObject(A_clmem);
	clStatus = clReleaseMemObject(B_clmem);
	clStatus = clReleaseMemObject(C_clmem);
	clStatus = clReleaseCommandQueue(command_queue);
	clStatus = clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	free(platforms);
	free(device_list);

	return 0;
}