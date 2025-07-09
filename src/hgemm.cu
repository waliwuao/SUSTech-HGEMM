#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <chrono>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

bool read_matrices_from_dir(const std::string& dir,
                            std::vector<__half>& A_fp16,
                            std::vector<__half>& B_fp16,
                            int& M, int& N, int& K) {
    std::string path_A = dir + "/A_matrix.bin";
    std::string path_B = dir + "/B_matrix.bin";

    std::ifstream fa(path_A, std::ios::binary);
    std::ifstream fb(path_B, std::ios::binary);
    if (!fa.is_open() || !fb.is_open()) {
        std::cerr << "Error opening binary matrix files in " << dir << std::endl;
        return false;
    }

    int m_a = 0, k_a = 0, k_b = 0, n_b = 0;

    fa.read(reinterpret_cast<char*>(&m_a), sizeof(int));
    fa.read(reinterpret_cast<char*>(&k_a), sizeof(int));
    size_t size_A = static_cast<size_t>(m_a) * k_a;
    A_fp16.resize(size_A);
    fa.read(reinterpret_cast<char*>(A_fp16.data()), size_A * sizeof(__half));

    fb.read(reinterpret_cast<char*>(&k_b), sizeof(int));
    fb.read(reinterpret_cast<char*>(&n_b), sizeof(int));
    size_t size_B = static_cast<size_t>(k_b) * n_b;
    B_fp16.resize(size_B);
    fb.read(reinterpret_cast<char*>(B_fp16.data()), size_B * sizeof(__half));

    fa.close();
    fb.close();

    if (k_a != k_b) {
        std::cerr << "Error: K dimension mismatch between A and B\n";
        return false;
    }

    M = m_a;
    K = k_a;
    N = n_b;
    return true;
}

__global__ void wmma_gemm_kernel_fp16(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    int M_pad = (M + WMMA_M - 1) / WMMA_M * WMMA_M;
    int N_pad = (N + WMMA_N - 1) / WMMA_N * WMMA_N;
    int K_pad = (K + WMMA_K - 1) / WMMA_K * WMMA_K;
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
    
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    for (int k = 0; k < K_pad; k += WMMA_K) {
        if (warpM * WMMA_M < M && k + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * K + k, K);
        } else {
            for (int i = 0; i < a_frag.num_elements; i++) {
                a_frag.x[i] = __float2half(0.0f);
            }
        }
        
        if (warpN * WMMA_N < N && k + WMMA_K <= K) {
            wmma::load_matrix_sync(b_frag, B + k * N + warpN * WMMA_N, N);
        } else {
            for (int i = 0; i < b_frag.num_elements; i++) {
                b_frag.x[i] = __float2half(0.0f);
            }
        }
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    if (warpM * WMMA_M < M && warpN * WMMA_N < N) {
        wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, c_frag, N, wmma::mem_row_major);
    }
}

int main(int argc, char* argv[]) {
    std::string input_dir = "data/input/Case1_768x768x768";
    std::string output_dir = "data/output";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--indir") && i + 1 < argc) {
            input_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--outdir") && i + 1 < argc) {
            output_dir = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [-d input_dir] [-o output_dir]" << std::endl;
            return 1;
        }
    }

    std::string case_name = input_dir.substr(input_dir.find_last_of("/\\") + 1);
    std::string output_file = output_dir + "/result_" + case_name + ".txt";

    int M, N, K;
    std::vector<__half> A_fp16, B_fp16;
    if (!read_matrices_from_dir(input_dir, A_fp16, B_fp16, M, N, K)) return 1;

    __half *d_A_fp16, *d_B_fp16, *d_C_custom;
    cudaMalloc(&d_A_fp16, M * K * sizeof(__half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(__half));
    cudaMalloc(&d_C_custom, M * N * sizeof(__half));

    cudaMemcpy(d_A_fp16, A_fp16.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, B_fp16.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C_custom, 0, M * N * sizeof(__half));
    cudaDeviceSynchronize();

    dim3 block(32, 4);
    dim3 grid((M + 15) / 16, (N + 15) / 16);

    auto start2 = std::chrono::high_resolution_clock::now();
    wmma_gemm_kernel_fp16<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_custom, M, N, K);
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();

    double duration_custom = std::chrono::duration<double, std::milli>(end2 - start2).count();

    std::vector<__half> C_custom_host(M * N);
    cudaMemcpy(C_custom_host.data(), d_C_custom, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    float sum_custom = 0.f;
    for (int i = 0; i < M * N; ++i) {
        sum_custom += __half2float(C_custom_host[i]);
    }

    double flops = 2.0 * M * N * K;
    double custom_gflops = flops / (duration_custom / 1000.0) / 1e9;

    std::cout << "WMMA FP16 Kernel Time: " << duration_custom << " ms, gFLOPS: " << custom_gflops << std::endl;
    std::cout << "WMMA Kernel Result sum: " << sum_custom << std::endl;

    std::ofstream out(output_file);
    if (out.is_open()) {
        out << "Case: " << case_name << "\n";
        out << "WMMA FP16 Kernel Time: " << duration_custom << " ms, gFLOPS: " << custom_gflops << "\n";
        out << "WMMA Kernel Result sum: " << sum_custom << "\n";
        out.close();
    }

    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_custom);

    return 0;
}
