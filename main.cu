#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <mpi.h>

__global__ void calc_borders(double *matrixA, double *matrixB, size_t size,
                             size_t sizePerGpu);

// The main calculation for our programm
__global__ void heat_equation(double *matrixA, double *matrixB, size_t size,
                              size_t sizePerGpu);

// difference of matrix
__global__ void get_error(double *matrixA, double *matrixB,
                          double *outputMatrix, size_t size,
                          size_t sizePerGpu);

int main(int argc, char **argv) {

  // Work with vars for our programm
  
  const double minError = atof(argv[1]);
  const int size = atoi(argv[2]);
  const int maxIter = atoi(argv[3]);
  const size_t totalSize = size * size;

  int rank, sizeOfTheGroup;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup);

  int numOfDevices = 0;
  cudaGetDeviceCount(&numOfDevices);

  cudaSetDevice(rank);

  // Board between devices ( I need to alocate some memory)
  size_t sizeOfAreaForOneProcess = size / sizeOfTheGroup;
  size_t startYIdx = sizeOfAreaForOneProcess * rank;

  // And alocation for them
  cudaMallocHost(&matrixA, sizeof(double) * totalSize);
  cudaMallocHost(&matrixB, sizeof(double) * totalSize);

  std::memset(matrixA, 0, size * size * sizeof(double));

  // Boards beetwen them
  matrixA[0] = 10;
  matrixA[size - 1] = 20;
  matrixA[size * size - 1] = 30;
  matrixA[size * (size - 1)] = 20;

  double step = 1.0 * (matrixA[size - 1] - matrixA[0]) / (size - 1);
  for (int i = 1; i < size - 1; i++) {
    matrixA[i] = matrixA[0] + i * step;
    matrixA[i * size] = matrixA[0] + i * step;
    matrixA[size - 1 + i * size] = matrixA[size - 1] + i * step;
    matrixA[size * (size - 1) + i] = matrixA[size * (size - 1)] + i * step;
  }

  std::memcpy(matrixB, matrixA, totalSize * sizeof(double));

  // Global vars for matrix
  double *matrixA = nullptr, *matrixB = nullptr, *deviceMatrixAPtr = nullptr,
         *deviceMatrixBPtr = nullptr, *deviceError = nullptr,
         *errorMatrix = nullptr, *tempStorage = nullptr;

  // Calculation for memory

  if (rank != 0 && rank != sizeOfTheGroup - 1) {
    sizeOfAreaForOneProcess += 2;
  } else {
    sizeOfAreaForOneProcess += 1;
  }

  size_t sizeOfAllocatedMemory = size * sizeOfAreaForOneProcess;

  unsigned int threads_x = std::min(size, 1024);
  unsigned int blocks_y = sizeOfAreaForOneProcess;
  unsigned int blocks_x = size / threads_x;

  dim3 blockDim(threads_x, 1);
  dim3 gridDim(blocks_x, blocks_y);

  ////////////////////////////////////////////////////////////////////////////
  // Here I give memory for one action( step or move)
  ////////////////////////////////////////////////////////////////////////////
  cudaMalloc((void **)&deviceMatrixAPtr,
             sizeOfAllocatedMemory * sizeof(double));
  cudaMalloc((void **)&deviceMatrixBPtr,
             sizeOfAllocatedMemory * sizeof(double));
  cudaMalloc((void **)&errorMatrix, sizeOfAllocatedMemory * sizeof(double));
  cudaMalloc((void **)&deviceError, sizeof(double));

  ////////////////////////////////////////////////////////////////////////////
  // Copy of matrix in located memory
  ////////////////////////////////////////////////////////////////////////////
  size_t offset = (rank != 0) ? size : 0;
  cudaMemcpy(deviceMatrixAPtr, matrixA + (startYIdx * size) - offset,
             sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMatrixBPtr, matrixB + (startYIdx * size) - offset,
             sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);

  ////////////////////////////////////////////////////////////////////////////
  // Work with buffer's size 
  ////////////////////////////////////////////////////////////////////////////
  size_t tempStorageSize = 0;
  cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError,
                         size * sizeOfAreaForOneProcess);
  cudaMalloc((void **)&tempStorage, tempStorageSize);

  double *error;
  cudaMallocHost(&error, sizeof(double));
  *error = 1.0;

  cudaStream_t stream, matrixCalculationStream;
  cudaStreamCreate(&stream);
  cudaStreamCreate(&matrixCalculationStream);

  int iter = 0;

  ////////////////////////////////////////////////////////////////////////////
  // start of calculation
  ////////////////////////////////////////////////////////////////////////////
  clock_t begin = clock();
  while ((iter < maxIter) && (*error) > minError) {
    iter++;

    ////////////////////////////////////////////////////////////////////////////
    // Calculation for boards ( I would like to send it for anothre process)
    ////////////////////////////////////////////////////////////////////////////
    calc_borders<<<size, 1, 0, stream>>>(deviceMatrixAPtr, deviceMatrixBPtr,
                                         size, sizeOfAreaForOneProcess);

    cudaStreamSynchronize(stream);
    ////////////////////////////////////////////////////////////////////////////
    // matrix calculation
    ////////////////////////////////////////////////////////////////////////////
    heat_equation<<<gridDim, blockDim, 0, matrixCalculationStream>>>(
        deviceMatrixAPtr, deviceMatrixBPtr, size, sizeOfAreaForOneProcess);

    ////////////////////////////////////////////////////////////////////////////
    // I check error for everu 100 iteration
    ////////////////////////////////////////////////////////////////////////////
    if (iter % 100 == 0) {
      get_error<<<gridDim, blockDim, 0, stream>>>(
          deviceMatrixAPtr, deviceMatrixBPtr, errorMatrix, size,
          sizeOfAreaForOneProcess);

      cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix,
                             deviceError, sizeOfAllocatedMemory, stream);

      cudaStreamSynchronize(stream);

      ////////////////////////////////////////////////////////////////////////////
      // In searching of max error
      ////////////////////////////////////////////////////////////////////////////
      MPI_Allreduce((void *)deviceError, (void *)deviceError, 1, MPI_DOUBLE,
                    MPI_MAX, MPI_COMM_WORLD);

      cudaMemcpyAsync(error, deviceError, sizeof(double),
                      cudaMemcpyDeviceToHost, stream);
    }

    // Work with boards (In searching of our boards)
    //   \
    //    |
    //    \/
    // Обмен верхней границей
    if (rank != 0) {
      MPI_Sendrecv(deviceMatrixBPtr + size + 1, size - 2, MPI_DOUBLE, rank - 1,
                   0, deviceMatrixBPtr + 1, size - 2, MPI_DOUBLE, rank - 1, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Обмен нижней границей
    if (rank != sizeOfTheGroup - 1) {
      MPI_Sendrecv(deviceMatrixBPtr + (sizeOfAreaForOneProcess - 2) * size + 1,
                   size - 2, MPI_DOUBLE, rank + 1, 0,
                   deviceMatrixBPtr + (sizeOfAreaForOneProcess - 1) * size + 1,
                   size - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    }

    cudaStreamSynchronize(matrixCalculationStream);
    // Rebuild for pointers
    double *temp = deviceMatrixAPtr;
    deviceMatrixAPtr = deviceMatrixBPtr;
    deviceMatrixBPtr = temp;
    //
    //
  }

  clock_t end = clock();
  if (rank == 0) {
    printf("Time is = %lf\n", 1.0 * (end - start) / CLOCKS_PER_SEC);
    printf("%d %lf\n", k, *error);
  }

  MPI_Finalize();

  cudaFree(deviceMatrixAPtr);
  cudaFree(deviceMatrixBPtr);
  cudaFree(errorMatrix);
  cudaFree(tempStorage);
  cudaFree(matrixA);
  cudaFree(matrixB);

  return 0;
}

__global__ void calc_borders(double *matrixA, double *matrixB, size_t size,
                             size_t sizePerGpu) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idxUp == 0 || idxUp > size - 2)
    return;

  if (idxUp < size) {
    matrixB[size + idx] =
        0.25 * (matrixA[size + idx - 1] + matrixA[2 * size + idx] +
                matrixA[*size + idx + 1]);
    matrixB[(sizePerGpu - 2) * size + idx] =
        0.25 * (matrixA[(sizePerGpu - 2) * size + idx - 1] +
                matrixA[(sizePerGpu - 3) * size + idx] +
                matrixA[(sizePerGpu - 1) * size + idx] +
                matrixA[(sizePerGpu - 2) * size + idx + 1]);
  }
}

__global__ void heat_equation(double *matrixA, double *matrixB, size_t size,
                              size_t sizePerGpu) {
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (!(j < 1 || i < 2 || j > size - 2 || i > sizePerGpu - 2)) {
    matrixB[i * size + j] =
        0.25 * (matrixA[(i + 1) * size + j] + matrixA[(i - 1) * size + j] +
                matrixA[i * size + (j + 1)] + matrixA[i * size + (j - 1)]);
  }
}

__global__ void get_error(double *matrixA, double *matrixB,
                          double *outputMatrix, size_t size,
                          size_t sizePerGpu) {
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

  size_t idx = i * size + j;
  if (!(j == 0 || i == 0 || j == size - 1 || i == sizePerGpu - 1)) {
    outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
  }
}