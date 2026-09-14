// Tiny MLP from scratch: 2 -> HIDDEN -> 1 (regression).
//
// Stack split (kept simple on purpose):
//   C++        : network definition, host memory (new/delete),
//                CSV I/O, training loop, SGD, backward pass on CPU.
//   CUDA C++   : forward pass on the GPU (2 kernels below),
//                device memory (cudaMalloc/cudaFree/cudaMemcpy).
//   Python     : dataset generation + evaluation/plotting (see *.py).
//
// Build:  sh build.sh        (or: nvcc -O2 -o build/mlp mlp.cu)
// Run:    ./build/mlp data_train.csv data_test.csv 500 0.05
// Output: losses.csv (epoch,loss), preds.csv (x1,x2,y_true,y_pred on test set)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------- GPU kernels
// Y = relu(X @ W + b).  X:[N,In] W:[In,Out] b:[Out] Y:[N,Out]. Row-major.
__global__ void fc_relu_forward(const float* X, const float* W, const float* b,
                                float* Y, int N, int In, int Out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Out) return;
    int row = idx / Out;
    int col = idx % Out;
    float s = b[col];
    for (int k = 0; k < In; ++k) s += X[row * In + k] * W[k * Out + col];
    Y[idx] = s > 0.0f ? s : 0.0f;
}

// Y = X @ W + b (linear output layer).
__global__ void fc_forward(const float* X, const float* W, const float* b,
                           float* Y, int N, int In, int Out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Out) return;
    int row = idx / Out;
    int col = idx % Out;
    float s = b[col];
    for (int k = 0; k < In; ++k) s += X[row * In + k] * W[k * Out + col];
    Y[idx] = s;
}

// ---------------------------------------------------------------- C++ helpers
static float rand_uniform(float lo, float hi) {
    return lo + (hi - lo) * (rand() / (float)RAND_MAX);
}

struct Dataset {
    int n = 0;                 // rows
    std::vector<float> X;      // [n*2]
    std::vector<float> y;      // [n]
};

static Dataset load_csv(const char* path) {
    Dataset d;
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        float x1, x2, y; char c;
        ss >> x1 >> c >> x2 >> c >> y;
        d.X.push_back(x1); d.X.push_back(x2); d.y.push_back(y);
        d.n++;
    }
    if (d.n == 0) { fprintf(stderr, "Empty dataset: %s\n", path); exit(1); }
    return d;
}

// ---------------------------------------------------------------- MLP network
// Host (C++) owns weights; device (GPU) runs the forward pass.
class MLP {
public:
    int in = 2, hid = 16, out = 1;
    float *W1 = nullptr, *b1 = nullptr;   // [in*hid], [hid]
    float *W2 = nullptr, *b2 = nullptr;   // [hid*out], [out]

    // Device buffers sized for the largest batch (train or test).
    float *d_X = nullptr, *d_W1 = nullptr, *d_b1 = nullptr;
    float *d_H = nullptr, *d_W2 = nullptr, *d_b2 = nullptr, *d_Y = nullptr;
    int dev_cap = 0; // max rows the device buffers hold

    explicit MLP(int hidden = 16) : hid(hidden) {
        srand(42);
        W1 = new float[in * hid]; b1 = new float[hid]();
        W2 = new float[hid * out]; b2 = new float[out]();
        float s1 = sqrtf(2.0f / in), s2 = sqrtf(2.0f / hid);
        for (int i = 0; i < in * hid; ++i) W1[i] = rand_uniform(-s1, s1);
        for (int i = 0; i < hid * out; ++i) W2[i] = rand_uniform(-s2, s2);
    }

    ~MLP() {
        delete[] W1; delete[] b1; delete[] W2; delete[] b2;
        cudaFree(d_X); cudaFree(d_W1); cudaFree(d_b1);
        cudaFree(d_H); cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_Y);
    }

    void alloc_device(int max_rows) {
        dev_cap = max_rows;
        CUDA_CHECK(cudaMalloc(&d_X,  max_rows * in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W1, in * hid * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b1, hid * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_H,  max_rows * hid * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W2, hid * out * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b2, out * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Y,  max_rows * out * sizeof(float)));
    }

    // Forward on GPU. Returns predictions in host vector `pred` ([N]).
    void forward_gpu(const float* Xh, int N, std::vector<float>& pred, std::vector<float>& H) {
        pred.resize(N); H.resize((size_t)N * hid);
        CUDA_CHECK(cudaMemcpy(d_X, Xh, (size_t)N * in * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W1, W1, in * hid * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, b1, hid * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2, W2, hid * out * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, b2, out * sizeof(float), cudaMemcpyHostToDevice));

        int t1 = N * hid, t2 = N * out;
        fc_relu_forward<<<(t1 + 255) / 256, 256>>>(d_X, d_W1, d_b1, d_H, N, in, hid);
        fc_forward<<<(t2 + 255) / 256, 256>>>(d_H, d_W2, d_b2, d_Y, N, hid, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(H.data(), d_H, (size_t)N * hid * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pred.data(), d_Y, (size_t)N * out * sizeof(float), cudaMemcpyDeviceToHost));
    }

    static float mse(const std::vector<float>& p, const std::vector<float>& t) {
        double s = 0;
        for (size_t i = 0; i < p.size(); ++i) { double d = p[i] - t[i]; s += d * d; }
        return (float)(s / p.size());
    }

    // One full-batch SGD step. Backward pass on CPU (tiny sizes, kept simple).
    float train_step(const Dataset& d, float lr) {
        int N = d.n;
        std::vector<float> pred, H;
        forward_gpu(d.X.data(), N, pred, H);

        // dOut = 2/N * (pred - y)
        std::vector<float> dOut(N);
        for (int i = 0; i < N; ++i) dOut[i] = 2.0f * (pred[i] - d.y[i]) / N;

        // Gradients of output layer: dW2 = H^T dOut, db2 = sum(dOut)
        std::vector<float> dW2(hid, 0.0f);
        float db2 = 0.0f;
        for (int i = 0; i < N; ++i) {
            db2 += dOut[i];
            for (int h = 0; h < hid; ++h) dW2[h] += H[(size_t)i * hid + h] * dOut[i];
        }
        // Backprop into hidden: dH = dOut * W2^T * relu'(H)
        std::vector<float> dH((size_t)N * hid);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hid; ++h) {
                float m = H[(size_t)i * hid + h] > 0.0f ? 1.0f : 0.0f;
                dH[(size_t)i * hid + h] = dOut[i] * W2[h] * m;
            }
        // Gradients of layer 1: dW1 = X^T dH, db1 = sum rows(dH)
        std::vector<float> dW1((size_t)in * hid, 0.0f);
        std::vector<float> db1(hid, 0.0f);
        for (int i = 0; i < N; ++i)
            for (int h = 0; h < hid; ++h) {
                db1[h] += dH[(size_t)i * hid + h];
                for (int k = 0; k < in; ++k)
                    dW1[(size_t)k * hid + h] += d.X[(size_t)i * in + k] * dH[(size_t)i * hid + h];
            }
        // SGD update
        for (int i = 0; i < in * hid; ++i) W1[i] -= lr * dW1[i];
        for (int i = 0; i < hid; ++i) b1[i] -= lr * db1[i];
        for (int i = 0; i < hid * out; ++i) W2[i] -= lr * dW2[i];
        b2[0] -= lr * db2;

        return mse(pred, d.y);
    }
};

int main(int argc, char** argv) {
    const char* train_path = argc > 1 ? argv[1] : "data_train.csv";
    const char* test_path  = argc > 2 ? argv[2] : "data_test.csv";
    int epochs = argc > 3 ? atoi(argv[3]) : 500;
    float lr   = argc > 4 ? (float)atof(argv[4]) : 0.05f;

    Dataset train = load_csv(train_path);
    Dataset test = load_csv(test_path);
    printf("Train: %d rows  Test: %d rows  epochs=%d lr=%g\n", train.n, test.n, epochs, lr);

    MLP net(16);
    net.alloc_device(train.n > test.n ? train.n : test.n);

    std::ofstream loss_f("losses.csv");
    loss_f << "epoch,loss\n";
    for (int e = 1; e <= epochs; ++e) {
        float loss = net.train_step(train, lr);
        if (e == 1 || e % 10 == 0 || e == epochs) {
            printf("epoch %4d/%d  mse=%.6f\n", e, epochs, loss);
            loss_f << e << "," << loss << "\n";
        }
    }
    loss_f.close();

    std::vector<float> pred, H;
    net.forward_gpu(test.X.data(), test.n, pred, H);
    std::ofstream pf("preds.csv");
    pf << "x1,x2,y_true,y_pred\n";
    for (int i = 0; i < test.n; ++i)
        pf << test.X[2*i] << "," << test.X[2*i+1] << "," << test.y[i] << "," << pred[i] << "\n";
    pf.close();
    printf("Test MSE: %.6f\nWrote losses.csv, preds.csv\n", MLP::mse(pred, test.y));
    return 0;
}
