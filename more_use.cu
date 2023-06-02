/*
 * Решение уравнения теплопроводности (пятиточечный шаблон) в двумерной области на равномерных
 * сетках (128^2, 256^2, 512^2, 1024^2). Граничные условия – линейная интерполяция между
 * углами области. Значение в углах – 10, 20, 30, 20.
 *
 * Параметры (точность, размер сетки, количество итераций) задаваются через
 * параметры командной строки.
 *
 * Вывод программы - количество итераций и достигнутое значение ошибки.
 *
 * Операция редукции (подсчет максимальной ошибки) для одного MPI процесса реализована с
 * использованием библиотеки CUB. Подсчет глобального значения ошибки, обмен граничными
 * условиями реализуется с использованием MPI
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include "cuda_runtime.h"
#include <cub/cub.cuh>
#include <mpi.h>

#ifdef _FLOAT
#define T float
#define MAX std::fmaxf
#define STOD std::stof
#define MPI_T MPI_FLOAT
#else
#define T double
#define MAX std::fmax
#define STOD std::stod
#define MPI_T MPI_DOUBLE
#endif

// Макрос индексации с 0
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))

// Макрос проверки статуса операции CUDA
#define CUDA_CHECK(err)                                                        \
    {                                                                          \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess)                                               \
        {                                                                      \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    }

// Вывести свойства девайса
void print_device_properties(void)
{
    cudaDeviceProp deviceProp;
    if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, 0))
    {
        printf("Warp size in threads is %d.\n", deviceProp.warpSize);
        printf("Maximum size of each dimension of a block is %d, %d, %d.\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid is %d, %d, %d.\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum resident threads per multiprocessor is %d.\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of resident blocks per multiprocessor is %d.\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("_____________________________________________________________________________________________\n");
    }
}

// Вывести значения двумерного массива на gpu
void print_array_gpu(T *A, uint32_t h, uint32_t w)
{
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
            printf("%.2f\t", A[IDX2C(i, j, w)]);
        printf("\n");
    }
    printf("\n");
}

// Инициализация матрицы, чтобы подготовить ее к основному алгоритму
void initialize_array(T *A, int size)
{
    // Заполнение углов матрицы значениями
    A[IDX2C(0, 0, size)] = 10.0;
    A[IDX2C(0, size - 1, size)] = 20.0;
    A[IDX2C(size - 1, 0, size)] = 20.0;
    A[IDX2C(size - 1, size - 1, size)] = 30.0;

    // Заполнение периметра матрицы
    T step = 10.0 / (size - 1);

    for (int i = 1; i < size - 1; ++i)
    {
        T addend = step * i;
        A[IDX2C(0, i, size)] = A[IDX2C(0, 0, size)] + addend;               // horizontal
        A[IDX2C(size - 1, i, size)] = A[IDX2C(size - 1, 0, size)] + addend; // horizontal
        A[IDX2C(i, 0, size)] = A[IDX2C(0, 0, size)] + addend;               // vertical
        A[IDX2C(i, size - 1, size)] = A[IDX2C(0, size - 1, size)] + addend; // vertical
    }
}

// Посчитать часть матрицы
__global__ void calculate_matrix(T *Anew, T *A, uint32_t h, uint32_t w)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    // Граница или выход за границы массива - ничего не делать
    if (i >= h - 1 || j >= w - 1 || i == 0 || j == 0)
        return;

    Anew[IDX2C(i, j, w)] = (A[IDX2C(i + 1, j, w)] + A[IDX2C(i - 1, j, w)] + A[IDX2C(i, j - 1, w)] + A[IDX2C(i, j + 1, w)]) * 0.25;
}

// O = |A-B|
__global__ void count_matrix_difference(T *matrixA, T *matrixB, T *outputMatrix, uint32_t h, uint32_t w)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    // Выход за границы массива или периметр - ничего не делать
    if (i >= h - 1 || j >= w - 1 || i == 0 || j == 0)
        return;

    uint32_t idx = IDX2C(i, j, w);
    outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
}

// Поделиться данными с соседями
void transfer_data(const int rank, const int size, T *A_host, T *A_device, uint32_t h, uint32_t w, cudaStream_t stream = 0)
{
    MPI_Status status;

    // Обмен с соседом сверху
    if (rank != 0)
    {
        // отправляем указатель на вторую строку процессу, работующему сверху, и принимаем строку от верхнего
        CUDA_CHECK(cudaMemcpyAsync(A_host + w, A_device + w, w * sizeof(T), cudaMemcpyDeviceToHost, stream));

        MPI_Sendrecv(A_host + w, w, MPI_T, rank - 1, rank - 1,
                               A_host, w, MPI_T, rank - 1, rank,
                               MPI_COMM_WORLD, &status);

        CUDA_CHECK(cudaMemcpyAsync(A_device, A_host, w * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    // Обмен с соседом снизу
    if (rank != size - 1)
    {
        // отправляем свою вторую строку вниз и принимаем строку от нижнего
        CUDA_CHECK(cudaMemcpyAsync(A_host + w * (h - 2), A_device + w * (h - 2), w * sizeof(T), cudaMemcpyDeviceToHost, stream));

        MPI_Sendrecv(A_host + w * (h - 2), w, MPI_T, rank + 1, rank + 1,
                               A_host + (w * (h - 1)), w, MPI_T, rank + 1, rank,
                               MPI_COMM_WORLD,&status);

        CUDA_CHECK(cudaMemcpyAsync(A_device + (w * (h - 1)), A_host + (w * (h - 1)), w * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
}

int main(int argc, char *argv[])
{
    // Парсинг аргументов командной строки
    int net_size = 128, iter_max = (int)1e6;
    T accuracy = 1e-6;
    bool res = false;
    for (int arg = 1; arg < argc; arg++)
    {
        std::string str = argv[arg];
        if (!str.compare("-res"))
            res = true;
        else
        {
            arg++;
            if (!str.compare("-a"))
                accuracy = STOD(argv[arg]);
            else if (!str.compare("-i"))
                iter_max = (int)std::stod(argv[arg]);
            else if (!str.compare("-s"))
                net_size = std::stoi(argv[arg]);
            else
            {
                std::cout << "Wrong args!\n";
                return -1;
            }
        }
    }

    // Инициализация библиотеки MPI
    MPI_Init(&argc, &argv);

    // Две информационных функции: сообщают размер группы (то есть общее количество задач, подсоединенных к ее области связи) и порядковый номер вызывающей задачи:
    //   MPI_COMM_WORLD - название коммуникатора (описателя области связи), создаваемого библиотекой автоматически.
    //   Он описывает стартовую область связи, объединяющую все процессы приложения.
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Размер сетки должен быть кратен количеству GPU
    if(net_size % size)
        MPI_Abort(MPI_COMM_WORLD, 5);

    // Каждый процесс выбирает свой девайс
    cudaSetDevice(rank);

    /*
     * rank 0 - главный процесс, на нем происходит инциализация и отправка частей массива другим процессам.
     * Каждый процесс обрабатывает net_size_local строк. Главный тоже рабоатет.
     * Матрица делится на равные части и эти части раздаются процессам, еще 2 строки нужны для
     * коммуникации с следующим процессом. Самые нижняя и самая верхняя строка будет восприниматься
     * процессом как граница и не будет изменена.
     */

    // Фактическое количество элементов в массиве
    const size_t vec_size_global = net_size * net_size;

    // Количество строк для данного процесса
    size_t net_height = net_size / size + 2 * (rank != size - 1); // Последнему не нужно еще 2 строки

    const size_t vec_size_local = net_height * net_size;

    // Объявление необходимых указателей и выделение памяти
    // Матрица на хосте (нужна только для инициализации и вывода) и
    // матрица ошибок на девайсе - сюда будут приходить по одному значению с каждого процесса [rank 0]
    T *A_global = nullptr, *Aerr_global = nullptr;

    // Матрицы для работы процесса: 1 на хосте, 2 на девайсе и еще одна на девайсе для разности
    T *A, *A_dev, *Anew_dev, *A_err_dev;

    CUDA_CHECK(cudaMallocHost(&A, sizeof(T) * vec_size_local));
    CUDA_CHECK(cudaMalloc(&A_dev, sizeof(T) * vec_size_local));
    CUDA_CHECK(cudaMalloc(&Anew_dev, sizeof(T) * vec_size_local));
    CUDA_CHECK(cudaMalloc(&A_err_dev, sizeof(T) * vec_size_local));

    if (rank == 0)
    {
        CUDA_CHECK(cudaMallocHost(&A_global, sizeof(T) * vec_size_global));
        CUDA_CHECK(cudaMallocHost(&Aerr_global, sizeof(T) * size));

        // Инициализация матрицы
        initialize_array(A_global, net_size);

        // if (res)
        //     print_array_gpu(A_global, net_size, net_size);

        // Отправка частей массива всем процессам
        // Смещение указателя начала данных
        int array_offset = 0;
        // Количество элементов для отправки
        int data_length = 0;
        for (size_t receiver = 0; receiver < size; ++receiver)
        {
            MPI_Request req;

            // Количество элементов для передачи. Как уже было отмечено,
            // последнему передается на 2 строки меньше, чем остальным.
            if (receiver == size - 1 && size > 1)
                data_length = (net_height - 2) * net_size;
            else
                data_length = vec_size_local;

            // Отправка
            MPI_Isend(A_global + array_offset, data_length, MPI_T, receiver, 0, MPI_COMM_WORLD, &req);

            // Не забываем, что есть еще 2 строки, используемые другими процессами
            array_offset += (net_height - 2) * net_size;
        }
    }

    // Получение данных от rank 0

    MPI_Status status; // А он зачем?
    MPI_Recv(A, vec_size_local, MPI_T, 0, 0, MPI_COMM_WORLD, &status);

    //std::cout << "rank = " << rank << "\tvec_size_local = " << vec_size_local << "\n";

    // Скопировать матрицу с хоста на матрицы на девайсе
    CUDA_CHECK(cudaMemcpy(A_dev, A, sizeof(T) * vec_size_local, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Anew_dev, A, sizeof(T) * vec_size_local, cudaMemcpyHostToDevice));

    // Потоков в одном блоке (32 * 32)
    dim3 threadPerBlock = dim3(32, 32);
    // Блоков в сетке (size / 32)
    dim3 blockPerGrid = dim3(ceil((double)net_height / 32), ceil((double)net_size / 32));
    // Текущая ошибка
    T *error, *error_dev;
    CUDA_CHECK(cudaMallocHost(&error, sizeof(T)));
    CUDA_CHECK(cudaMalloc(&error_dev, sizeof(T)));
    *error = accuracy + 1;

    // Временный буфер для редукции и его размер
    T *reduction_bufer = NULL;
    uint64_t reduction_bufer_size = 0;

    // Первый вызов, чтобы предоставить количество байтов, необходимое для временного хранения, необходимого CUB.
    cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err_dev, error_dev, vec_size_local);

    // Выделение памяти под буфер
    CUDA_CHECK(cudaMalloc(&reduction_bufer, reduction_bufer_size));

    // Сокращение количества обращений к CPU. Больше сетка - реже стоит проверять ошибку.
    uint32_t skipped_checks_counts = (iter_max < net_size) ? iter_max : net_size;
    skipped_checks_counts += skipped_checks_counts % 2; // Привести к четному числу

    // Счетчик итераций
    int iter = 0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Начать отсчет времени работы
    auto begin_main = std::chrono::steady_clock::now();

    for (iter = 0; iter < iter_max && *error > accuracy; iter += skipped_checks_counts)
    {
        for (uint32_t k = 0; k < skipped_checks_counts; k += 2)
        {
            // Итерация пересчета
            calculate_matrix<<<blockPerGrid, threadPerBlock, 0, stream>>>(A_dev, Anew_dev, net_height, net_size);
            // Обмен граничными условиями
            transfer_data(rank, size, A, A_dev, net_height, net_size, stream);

            // То же самое, но матрицы A_dev и Anew_dev поменялись местами (замена swap)
            calculate_matrix<<<blockPerGrid, threadPerBlock, 0, stream>>>(Anew_dev, A_dev, net_height, net_size);
            transfer_data(rank, size, A, Anew_dev, net_height, net_size, stream);
        }

        count_matrix_difference<<<blockPerGrid, threadPerBlock, 0, stream>>>(A_dev, Anew_dev, A_err_dev, net_height, net_size);

        // Найти максимум и положить в error_dev - аналог reduction (max : error_dev) в OpenACC
        cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err_dev, error_dev, vec_size_local, stream);

        // Копировать ошибку с девайса на хост
        CUDA_CHECK(cudaMemcpy(error, error_dev, sizeof(T), cudaMemcpyDeviceToHost));

        // Сборка локальных максимальных ошибок с каждого процесса
        MPI_Gather(error, 1, MPI_T, Aerr_global, 1, MPI_T, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            *error = 0;
            for (int i = 0; i < size; ++i)
            {
                *error = MAX(*error, Aerr_global[i]);
            }
        }
        // Точка синхронизации
        MPI_Barrier(MPI_COMM_WORLD);
        // Разослать всем процессам глобальную максимальную ошибку
        MPI_Bcast(error, 1, MPI_T, 0, MPI_COMM_WORLD);
    }
    cudaStreamDestroy(stream);

    // Посчитать время выполнения
    auto end_main = std::chrono::steady_clock::now();
    int time_spent = std::chrono::duration_cast<std::chrono::milliseconds>(end_main - begin_main).count();
    // Вывод
    if (res)
    {
        // Отправка на хост
        CUDA_CHECK(cudaMemcpy(A, A_dev, sizeof(T) * vec_size_local, cudaMemcpyDeviceToHost));

        // Отправка rank 0
        MPI_Request req;
        // Первая и последняя строка пропускается
        MPI_Isend(A + net_size, (net_height - 2) * net_size, MPI_T, 0, 0, MPI_COMM_WORLD, &req);

        if (rank == 0)
        {
            // Начинаем принимать со второй строки
            int array_offset = net_size;
            MPI_Status status;
            for (size_t sender = 0; sender < size; ++sender)
            {
                int recive_size = 0;
                if (sender != size - 1 || size == 1)
                    recive_size = (net_height - 2) * net_size;
                else
                    recive_size = (net_height - 4) * net_size;

                MPI_Recv(A_global + array_offset, recive_size, MPI_T, sender, 0, MPI_COMM_WORLD, &status);
                array_offset += recive_size;
            }

            // И наконец-то вывод в консоль
            print_array_gpu(A_global, net_size, net_size);
        }
    }
    if (rank == 0)
    {
        std::cout << "Iter: " << iter << " Error: " << *error << std::endl;
        std::cout << "Time:\t\t" << time_spent << " ms\n";
        CUDA_CHECK(cudaFreeHost(A_global));
        CUDA_CHECK(cudaFreeHost(Aerr_global));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Освобождение памяти
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(error));
    CUDA_CHECK(cudaFree(reduction_bufer));
    CUDA_CHECK(cudaFree(A_err_dev));
    CUDA_CHECK(cudaFree(A_dev));
    CUDA_CHECK(cudaFree(Anew_dev));
    CUDA_CHECK(cudaFree(error_dev));

    // Нормальное закрытие библиотеки
    MPI_Finalize();
    return 0;
}
