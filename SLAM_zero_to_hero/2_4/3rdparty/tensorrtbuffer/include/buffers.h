/*
 * TensorRT Buffer Management Utilities
 *
 * This header provides utilities for managing GPU/CPU buffers used in
 * TensorRT inference. It simplifies memory allocation and data transfer
 * between host and device.
 *
 * Based on NVIDIA TensorRT samples.
 */

#ifndef TENSORRT_BUFFERS_H
#define TENSORRT_BUFFERS_H

#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace tensorrt_buffer {

/**
 * @brief RAII wrapper for GPU memory allocation
 */
class DeviceBuffer {
public:
    DeviceBuffer() : mBuffer(nullptr), mSize(0) {}

    DeviceBuffer(size_t size) : mSize(size) {
        if (size > 0) {
            cudaMalloc(&mBuffer, size);
        }
    }

    ~DeviceBuffer() {
        if (mBuffer) {
            cudaFree(mBuffer);
        }
    }

    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Enable move
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : mBuffer(other.mBuffer), mSize(other.mSize) {
        other.mBuffer = nullptr;
        other.mSize = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (mBuffer) cudaFree(mBuffer);
            mBuffer = other.mBuffer;
            mSize = other.mSize;
            other.mBuffer = nullptr;
            other.mSize = 0;
        }
        return *this;
    }

    void* data() { return mBuffer; }
    const void* data() const { return mBuffer; }
    size_t size() const { return mSize; }

private:
    void* mBuffer;
    size_t mSize;
};

/**
 * @brief RAII wrapper for pinned (page-locked) host memory
 *
 * Pinned memory enables faster CPU-GPU transfers.
 */
class HostBuffer {
public:
    HostBuffer() : mBuffer(nullptr), mSize(0) {}

    HostBuffer(size_t size) : mSize(size) {
        if (size > 0) {
            cudaMallocHost(&mBuffer, size);
        }
    }

    ~HostBuffer() {
        if (mBuffer) {
            cudaFreeHost(mBuffer);
        }
    }

    // Disable copy
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    // Enable move
    HostBuffer(HostBuffer&& other) noexcept
        : mBuffer(other.mBuffer), mSize(other.mSize) {
        other.mBuffer = nullptr;
        other.mSize = 0;
    }

    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this != &other) {
            if (mBuffer) cudaFreeHost(mBuffer);
            mBuffer = other.mBuffer;
            mSize = other.mSize;
            other.mBuffer = nullptr;
            other.mSize = 0;
        }
        return *this;
    }

    void* data() { return mBuffer; }
    const void* data() const { return mBuffer; }
    size_t size() const { return mSize; }

private:
    void* mBuffer;
    size_t mSize;
};

/**
 * @brief Manages host and device buffers for TensorRT inference
 *
 * This class handles:
 * - Allocating GPU memory for inputs/outputs
 * - Allocating pinned host memory for data transfer
 * - Copying data between host and device
 */
class BufferManager {
public:
    /**
     * @brief Create buffer manager for engine
     *
     * @param engine TensorRT engine
     * @param batchSize Batch size (usually 1)
     * @param context Execution context (for dynamic shapes)
     */
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                  int batchSize = 0,
                  nvinfer1::IExecutionContext* context = nullptr)
        : mEngine(engine), mBatchSize(batchSize) {

        // Allocate buffers for all bindings
        for (int i = 0; i < mEngine->getNbBindings(); ++i) {
            auto dims = context ? context->getBindingDimensions(i)
                               : mEngine->getBindingDimensions(i);

            size_t vol = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                vol *= dims.d[j] > 0 ? dims.d[j] : 1;
            }

            size_t elemSize = sizeof(float);  // Assume float32

            mHostBuffers.emplace_back(vol * elemSize);
            mDeviceBuffers.emplace_back(vol * elemSize);
            mDeviceBindings.push_back(mDeviceBuffers.back().data());
        }
    }

    /**
     * @brief Get host buffer by binding name
     */
    void* getHostBuffer(const std::string& tensorName) {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index < 0) {
            std::cerr << "Invalid tensor name: " << tensorName << std::endl;
            return nullptr;
        }
        return mHostBuffers[index].data();
    }

    /**
     * @brief Get device buffer by binding name
     */
    void* getDeviceBuffer(const std::string& tensorName) {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index < 0) return nullptr;
        return mDeviceBuffers[index].data();
    }

    /**
     * @brief Get array of device buffer pointers for executeV2
     */
    std::vector<void*>& getDeviceBindings() {
        return mDeviceBindings;
    }

    /**
     * @brief Copy all input buffers from host to device
     */
    void copyInputToDevice() {
        for (int i = 0; i < mEngine->getNbBindings(); ++i) {
            if (mEngine->bindingIsInput(i)) {
                cudaMemcpy(mDeviceBuffers[i].data(),
                          mHostBuffers[i].data(),
                          mHostBuffers[i].size(),
                          cudaMemcpyHostToDevice);
            }
        }
    }

    /**
     * @brief Copy all output buffers from device to host
     */
    void copyOutputToHost() {
        for (int i = 0; i < mEngine->getNbBindings(); ++i) {
            if (!mEngine->bindingIsInput(i)) {
                cudaMemcpy(mHostBuffers[i].data(),
                          mDeviceBuffers[i].data(),
                          mDeviceBuffers[i].size(),
                          cudaMemcpyDeviceToHost);
            }
        }
    }

    /**
     * @brief Copy all input buffers asynchronously
     */
    void copyInputToDeviceAsync(cudaStream_t stream) {
        for (int i = 0; i < mEngine->getNbBindings(); ++i) {
            if (mEngine->bindingIsInput(i)) {
                cudaMemcpyAsync(mDeviceBuffers[i].data(),
                               mHostBuffers[i].data(),
                               mHostBuffers[i].size(),
                               cudaMemcpyHostToDevice,
                               stream);
            }
        }
    }

    /**
     * @brief Copy all output buffers asynchronously
     */
    void copyOutputToHostAsync(cudaStream_t stream) {
        for (int i = 0; i < mEngine->getNbBindings(); ++i) {
            if (!mEngine->bindingIsInput(i)) {
                cudaMemcpyAsync(mHostBuffers[i].data(),
                               mDeviceBuffers[i].data(),
                               mDeviceBuffers[i].size(),
                               cudaMemcpyDeviceToHost,
                               stream);
            }
        }
    }

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    int mBatchSize;
    std::vector<HostBuffer> mHostBuffers;
    std::vector<DeviceBuffer> mDeviceBuffers;
    std::vector<void*> mDeviceBindings;
};

}  // namespace tensorrt_buffer

#endif  // TENSORRT_BUFFERS_H
