/*
 * TensorRT Common Utilities
 *
 * Provides common utilities for TensorRT applications including
 * smart pointer helpers, CUDA stream management, and DLA support.
 *
 * Based on NVIDIA TensorRT samples.
 */

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

#include <memory>
#include <cassert>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

// Assertion macro
#define ASSERT(condition)                                                       \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "Assertion failed: " #condition << " at " << __FILE__ \
                      << ":" << __LINE__ << std::endl;                         \
            abort();                                                           \
        }                                                                       \
    } while (0)

namespace tensorrt_common {

/**
 * @brief Custom deleter for TensorRT objects
 *
 * TensorRT objects need explicit destruction (they don't have virtual destructors
 * in older versions), so we use a custom deleter with unique_ptr.
 */
struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

/**
 * @brief Smart pointer type for TensorRT objects
 */
template <typename T>
using TensorRTUniquePtr = std::unique_ptr<T, InferDeleter>;

/**
 * @brief RAII wrapper for CUDA stream
 */
class CudaStream {
public:
    CudaStream() {
        cudaStreamCreate(&mStream);
    }

    ~CudaStream() {
        cudaStreamDestroy(mStream);
    }

    cudaStream_t get() const { return mStream; }
    operator cudaStream_t() const { return mStream; }

private:
    cudaStream_t mStream;
};

/**
 * @brief Create a CUDA stream wrapped in unique_ptr
 */
inline std::unique_ptr<CudaStream> makeCudaStream() {
    return std::make_unique<CudaStream>();
}

/**
 * @brief Memory size literal (MiB)
 *
 * Usage: config->setMaxWorkspaceSize(512_MiB);
 */
constexpr long long int operator"" _MiB(unsigned long long val) {
    return val * (1 << 20);
}

/**
 * @brief Memory size literal (GiB)
 *
 * Usage: config->setMaxWorkspaceSize(1_GiB);
 */
constexpr long long int operator"" _GiB(unsigned long long val) {
    return val * (1 << 30);
}

/**
 * @brief Enable Deep Learning Accelerator (DLA) on Jetson
 *
 * @param builder TensorRT builder
 * @param config Builder configuration
 * @param dlaCore DLA core index (-1 to disable)
 */
inline void enableDLA(nvinfer1::IBuilder* builder,
                      nvinfer1::IBuilderConfig* config,
                      int dlaCore) {
    if (dlaCore < 0) {
        return;  // DLA disabled
    }

    if (builder->getNbDLACores() == 0) {
        std::cerr << "Warning: No DLA cores available, running on GPU." << std::endl;
        return;
    }

    if (dlaCore >= builder->getNbDLACores()) {
        std::cerr << "Warning: DLA core " << dlaCore << " not available, "
                  << "only " << builder->getNbDLACores() << " cores present. "
                  << "Running on GPU." << std::endl;
        return;
    }

    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(dlaCore);

    // Allow GPU fallback for unsupported layers
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

    std::cout << "Using DLA core " << dlaCore << std::endl;
}

/**
 * @brief Get element size for TensorRT data type
 */
inline size_t getElementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 0;
    }
}

/**
 * @brief Calculate volume (total elements) of a tensor
 */
inline int64_t volume(const nvinfer1::Dims& dims) {
    int64_t vol = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        vol *= dims.d[i];
    }
    return vol;
}

}  // namespace tensorrt_common

#endif  // TENSORRT_COMMON_H
