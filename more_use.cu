#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <exception>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

// error checks
#define CUDACHECK(name) if (cudaGetLastError() != cudaSuccess) { throw std::runtime_error(name); } 
#define MPI_CHECK(code, name) if (code != MPI_SUCCESS) { throw std::runtime_error(name); }

// macros for average interpolation calculation
#define AVG_CALC(A, Anew, size, i, j) Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);

// pointers for error and other matrixes
double 	*A 		= nullptr,  // buffer for main matrix
	*Anew		= nullptr,  // buffer for matrix where we store our interpolations
	*dev_A 	        = nullptr,  // A on device
	*dev_Anew	= nullptr,  // Anew on device
        *buff           = nullptr,  // buffer for abs_diff calculation
	*d_out 		= nullptr,  // buffer for error on device
	*d_temp_storage = nullptr;  // temporary buffer for cub max reduction

// handler funnction which executes before end of program execution and frees memory allocated dynamically
void free_pointers()
{
	std::cout << "End of execution" << std::endl;
	
	// free memory section
	if (A) 	            cudaFreeHost(A); 		CUDACHECK("free A")
	if (Anew) 	    cudaFreeHost(Anew); 	CUDACHECK("free Anew")
	if (dev_A)	    cudaFree(dev_A); 		CUDACHECK("free dev_A")
	if (dev_Anew) 	    cudaFree(dev_Anew); 	CUDACHECK("free dev_Anew")
    	if (buff)           cudaFree(buff); 		CUDACHECK("free buff")
	if (d_out) 	    cudaFree(d_out);	        CUDACHECK("free d_out")
	if (d_temp_storage) cudaFree(d_temp_storage);   CUDACHECK("free d_temp_storage")
		
	std::cout << "Memory has been freed" << std::endl;
}

// interpolation on matrix field
__global__ void interpolate(double* A, double* Anew, size_t size, size_t size_per_one_gpu)
{
	unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

	if(!(x_idx < 1 || y_idx < 2 || x_idx > size - 2 || y_idx > size_per_one_gpu - 2)) {
		AVG_CALC(A, Anew, size, y_idx, x_idx)
	}	
}

// interpolation on the matrix edges between devices
__global__ void interpolate_boundaries(double* A, double* Anew, size_t size, size_t size_per_one_gpu){
	unsigned int up_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int down_idx = blockIdx.x * blockDim.x + threadIdx.x;

	// check if horizontal index between 1 and (size - 2) then calculates result
	if (!(up_idx == 0 || up_idx > size - 2)) {
		AVG_CALC(A, Anew, size, 1, up_idx)
		AVG_CALC(A, Anew, size, (size_per_one_gpu - 2), down_idx)
	}
}

// modular difference between A and Anew stored in buff
__global__ void abs_diff(double* A, double* Anew, double* buff, size_t size, size_t size_per_one_gpu) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx = y * size + x;
	
	// check if idx in allocated area then calculate result
	if(!(x <= 0 || y <= 0 || x >= (size - 1) || y >= (size_per_one_gpu - 1)))
	{
		buff[idx] = std::abs(A[idx] - Anew[idx]);
	}
}

int findNearestPowerOfTwo(size_t num) {
    int power = 1;
    while (power < num) {
        power <<= 1;
    }
    return power;
}

int main(int argc, char* argv[])
{
	auto atExifStatus = std::atexit(free_pointers);
	if (atExifStatus != 0)
	{
		std::cout << "Register error" << std::endl;
		exit(-1);
	}

	if (argc != 4)
	{
		std::cout << "Invalid parameters count" << std::endl;
		std::exit(-1);
	}
	
	try {
		//reads command prompt arguments: ./task4.out [max_aaccuracy] [size] [max_iterations]
		double max_accuracy = std::stod(argv[1]);
		int size = std::stoi(argv[2]);
		int matrixSize = size * size;  // total matrix size
		int max_iterations = std::stoi(argv[3]);

		// rank - number of device, device_group_size - number of devices used by MPI, error_code - buffer for error message
		int rank, device_group_size, error_code;

		// init MPI from command prompt arguments
		error_code = MPI_Init(&argc, &argv);
		MPI_CHECK(error_code, "mpi initialization")

		// get commutator rank from all possible
		error_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_CHECK(error_code, "mpi communicator rank_init")

		// get device count used by MPI
		error_code = MPI_Comm_size(MPI_COMM_WORLD, &device_group_size);
		MPI_CHECK(error_code, "mpi communicator size_init")

		// check if programm uses enough number of devices for next calculations
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		printf("%d - number of devices\n", deviceCount);
		if (deviceCount < device_group_size || device_group_size < 1) {
			std::cout << "INvalid number of devices!";
			std::exit(-1);
		}

		// choose device
		cudaSetDevice(rank);
		CUDACHECK("cuda set device")
		printf("device rank: %d\n", rank);

		// edges for calculating
		size_t area_for_one_process = size / device_group_size;
		size_t start_y_idx = area_for_one_process * rank;

		//allocate matrixes on host
		cudaMallocHost((void**)&A, matrixSize * sizeof(double));
		CUDACHECK("A host alloc")
		cudaMallocHost((void**)&Anew, matrixSize * sizeof(double));
		CUDACHECK("Anew host alloc")

		std::memset(A, 0, matrixSize * sizeof(double));
		std::memset(Anew, 0, matrixSize * sizeof(double));


		// matrix edge interpolation
		A[0] = 10.0;
		A[size - 1] = 20.0;
		A[size * size - 1] = 30.0;
		A[size * (size - 1)] = 20.0;

		Anew[0] = 10.0;
		Anew[size - 1] = 20.0;
		Anew[size * size - 1] = 30.0;
		Anew[size * (size - 1)] = 20.0;

		double step = 10.0 / (size - 1);
		for (int i = 1; i < size - 1; i++) {
			A[i] = A[0] + i * step;
			A[i * size] = A[0] + i * step;
			A[size - 1 + size * i] = A[size - 1] + i * step;
			A[size * (size - 1) + i] = A[size * (size - 1)] + i * step;

			Anew[i] = Anew[0] + i * step;
			Anew[i * size] = Anew[0] + i * step;
			Anew[size - 1 + size * i] = Anew[size - 1] + i * step;
			Anew[size * (size - 1) + i] = Anew[size * (size - 1)] + i * step;
		}

		// calculate used area for each process
		if (rank != 0 && rank != device_group_size - 1)
		{
			area_for_one_process += 2;
		}
		else if (device_group_size != 1)
		{
			area_for_one_process += 1;
		}

		// memory size for one device
		size_t alloc_memsize = size * area_for_one_process;

		// memory allocation for pointer will be used on device
		cudaMalloc((void**)&buff, alloc_memsize * sizeof(double));
		CUDACHECK("alloc buff")
		cudaMalloc((void**)&dev_A, alloc_memsize * sizeof(double));
		CUDACHECK("alloc dev_A")
		cudaMalloc((void**)&dev_Anew, alloc_memsize * sizeof(double));
		CUDACHECK("alloc dev_Anew")

		// memset + memcpy
		size_t offset = (rank != 0) ? size : 0;
		cudaMemset(dev_A, 0, sizeof(double) * alloc_memsize);
		CUDACHECK("memset dev_A")
		cudaMemset(dev_Anew, 0, sizeof(double) * alloc_memsize);
		CUDACHECK("memset dev_Anew")
		cudaMemcpy(dev_A, A + (start_y_idx * size) - offset, sizeof(double) * alloc_memsize, cudaMemcpyHostToDevice);
		CUDACHECK("memcpy from A to dev_A from start_y_idx coordinate with offset")
		cudaMemcpy(dev_Anew, Anew + (start_y_idx * size) - offset, sizeof(double) * alloc_memsize, cudaMemcpyHostToDevice);
		CUDACHECK("memcpy from Anew to dev_Anew with from start_y_idx coordinate offset")

		// allocates buffer 'd_out' to contain max('abs_diff' function result)
		double* d_out;
		cudaMalloc((void**)&d_out, sizeof(double));
		CUDACHECK("alloc d_out")

		// allocates memory for temporary storage ton use Max reduction and sets temp_storage_bytes with size of d_temp_storage in bytes
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, size * area_for_one_process);
		CUDACHECK("get temp_storage_bytes")
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		CUDACHECK("temp storage memory allocation")

		// variables for loop execution
		double accuracy = max_accuracy + 1.0;  // current accuracy
		int num_of_iterations = 0;  // current number of iterations

		// streams for calculations: cuda_stream - for blocks to sync them, matrix_calc_stream - for othre operation
		cudaStream_t cuda_stream, matrix_calc_stream;
		cudaStreamCreate(&cuda_stream);
		CUDACHECK("cuda_stream creation")
		cudaStreamCreate(&matrix_calc_stream);
		CUDACHECK("matrix_calc_stream creation")

		// params for cuda functions
		unsigned int threads_x = std::min(findNearestPowerOfTwo(size), 1024);
		unsigned int blocks_y = area_for_one_process;
		unsigned int blocks_x = size / threads_x + 1;

		dim3 blockDim(threads_x, 1);
		dim3 gridDim(blocks_x, blocks_y);

		while (num_of_iterations < max_iterations && accuracy > max_accuracy) {

			interpolate_boundaries<<<size, 1, 0, cuda_stream>>>(dev_A, dev_Anew, size, area_for_one_process);

			interpolate<<<gridDim, blockDim, 0, matrix_calc_stream>>>(dev_A, dev_Anew, size, area_for_one_process);
			
			// updates accuracy 1/100 times of main cycle iterations and on the last iteration
			if (num_of_iterations % 100 == 0 || num_of_iterations + 1 == max_iterations) {
				
				// synchronize to understand either we can make operations with matrix or not
				cudaStreamSynchronize(cuda_stream);
				CUDACHECK("cuda_stream synchronize after interpolation")

				abs_diff<<<gridDim, blockDim, 0, matrix_calc_stream>>>(dev_A, dev_Anew, buff, size, area_for_one_process);

				// cub max reduction
				cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, alloc_memsize, matrix_calc_stream);
				CUDACHECK("cub max reduction")

				// synchronize streams to receive actual d_out max values from all devices
				cudaStreamSynchronize(matrix_calc_stream);
				CUDACHECK("matrix_calc_stream synchronization (inside error calculations)")

				// receive max d_out values from all devices
				error_code = MPI_Allreduce((void*)d_out, (void*)d_out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				MPI_CHECK(error_code, "mpi reduction")

				// copy values from d_out on GPU to accuracy on CPU
				cudaMemcpyAsync(&accuracy, d_out, sizeof(double), cudaMemcpyDeviceToHost, matrix_calc_stream);
				CUDACHECK("copy to accuracy")
			}

			// receive top edge
			if (rank != 0)
			{
				error_code = MPI_Sendrecv(
					dev_Anew + size + 1,  // sending buffer
					size - 2,  // size of sent data
					MPI_DOUBLE,  // sening data type
					rank - 1,  // where data was sent
					0,  //
					dev_Anew + 1,  // receiving buffer
					size - 2,  // size of received data
					MPI_DOUBLE,  // received data type
					rank - 1,  // received from which device
					0, 
					MPI_COMM_WORLD,  // communicator
					MPI_STATUS_IGNORE
				);
				MPI_CHECK(error_code, "top edge receiving")
			}

			// receive bottom edge
			if (rank != device_group_size - 1)
			{
				error_code = MPI_Sendrecv(
					dev_Anew + (area_for_one_process - 2) * size + 1, 
					size - 2, 
					MPI_DOUBLE, 
					rank + 1,
					0,
					dev_Anew + (area_for_one_process - 1) * size + 1, 
					size - 2, 
					MPI_DOUBLE, 
					rank + 1, 
					0, 
					MPI_COMM_WORLD, 
					MPI_STATUS_IGNORE
				);
				MPI_CHECK(error_code, "bottom edge receiving")
			}

			// synchronize streams before next calculations
			cudaStreamSynchronize(matrix_calc_stream);
			CUDACHECK("matrix_calc_stream synchronization (main loop after MPI_Sendrecv)")

			++num_of_iterations;
			std::swap(dev_A, dev_Anew); // swap pointers for next calculations
		}

		cudaStreamDestroy(cuda_stream);
		CUDACHECK("Destroy cuda_stream")
		cudaStreamDestroy(matrix_calc_stream);
		CUDACHECK("Destroy matrix_calc_stream")

		if (rank == 0) {
			printf("Iterations: %d\nAccuracy: %lf\n", num_of_iterations, accuracy);
		}
		
		// end MPI engine
		error_code = MPI_Finalize();
		MPI_CHECK(error_code, "mpi finalize")

		std::cout << "MPI engine was shut down" << std::endl;

	}
	catch (std::runtime_error& error) {
		std::cout << error.what() << std::endl;
		std::cout << "Program execution stops" << std::endl;
		exit(-1);	
	}

	return 0;
}