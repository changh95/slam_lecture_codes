/* Copyright (c) 2019 Xieyuanli Chen, Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */
#pragma once

// For plugin factory
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include <chrono>
#include <numeric>
#include "net.hpp"

#define MAX_WORKSPACE_SIZE \
  (1UL << 32)  // gpu workspace size
#define MIN_WORKSPACE_SIZE (1UL << 20)  // gpu workspace size (pretty bad)

#define DEVICE_DLA_0 0  // jetson DLA 0 enabled
#define DEVICE_DLA_1 0  // jetson DLA 1 enabled

using namespace nvinfer1;  // I'm taking a liberty because the code is
                           // unreadable otherwise

#define CUDA_CHECK(status)                                             \
  {                                                                    \
    if (status != cudaSuccess) {                                       \
      printf("%s in %s at %d\n", cudaGetErrorString(status), __FILE__, \
             __LINE__);                                                \
      exit(-1);                                                        \
    }                                                                  \
  }

namespace rangenet {
namespace segmentation {

// Logger for GIE info/warning/errors
class Logger : public ILogger {
 public:
  void set_verbosity(bool verbose) { _verbose = verbose; }
  void log(Severity severity, const char* msg) noexcept override {
    if (_verbose) {
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          std::cerr << "INTERNAL_ERROR: ";
          break;
        case Severity::kERROR:
          std::cerr << "ERROR: ";
          break;
        case Severity::kWARNING:
          std::cerr << "WARNING: ";
          break;
        case Severity::kINFO:
          std::cerr << "INFO: ";
          break;
        default:
          std::cerr << "UNKNOWN: ";
          break;
      }
      std::cout << msg << std::endl;
    }
  }

 private:
  bool _verbose = false;
};

/**
 * @brief      Class for segmentation network inference with TensorRT.
 */
class NetTensorRT : public Net {
 public:
  /**
   * @brief      Constructs the object.
   *
   * @param[in]  model_path  The model path for the inference model directory
   *                         containing the "model.onnx" or "model.trt" file and the arch_cfg, data_cfg
   */
  NetTensorRT(const std::string& model_path);

  /**
   * @brief      Destroys the object.
   */
  ~NetTensorRT();

  /**
   * @brief      argsort.
   *
   * @param[in]  std::vector<T>
   *
   * @return     argsorted idxes
   */
  template <typename T>
  std::vector<size_t> sort_indexes(const std::vector<T> &v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v. >: decrease <: increase
    std::sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
  }


  /**
   * @brief      Project a pointcloud into a spherical projection image.projection.
   *
   * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
   *
   * @return     Projected LiDAR scans, with size of (_img_h * _img_w, _img_d)
   */
  std::vector<std::vector<float> > doProjection(const std::vector<float>& scan, const uint32_t& num_points);

 /**
  * @brief      Infer logits from LiDAR scan
  *
  * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
  *
  * @return     Semantic estimates with probabilities over all classes (_n_classes, _img_h, _img_w)
  */
  std::vector<std::vector<float> > infer(const std::vector<float>& scan, const uint32_t& num_points);

  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  void verbosity(const bool verbose);

  /**
   * @brief Get the Buffer Size object
   *
   * @param d dimension
   * @param t data type
   * @return int size of data
   */
  size_t getBufferSize(Dims d, DataType t);

  /**
   * @brief Deserialize an engine that comes from a previous run
   *
   * @param engine_path
   */
  void deserializeEngine(const std::string& engine_path);

  /**
   * @brief Serialize an engine that we generated in this run
   *
   * @param engine_path
   */
  void serializeEngine(const std::string& engine_path);

  /**
   * @brief Generate an engine from ONNX model
   *
   * @param onnx_path path to onnx file
   */
  void generateEngine(const std::string& onnx_path);

  /**
   * @brief Prepare io buffers for inference with engine
   */
  void prepareBuffer();


 protected:
  IRuntime* _runtime;    // TensorRT 10: runtime must outlive the engine
  ICudaEngine* _engine;  // tensorrt engine
  IExecutionContext* _context; // execution context (must destroy in destructor too)
  Logger _gLogger;  // trt logger
  std::vector<void*> _deviceBuffers;  // device mem
  cudaStream_t _cudaStream;           // cuda stream for async ops
  std::vector<void*> _hostBuffers;
  uint _inBindIdx;
  uint _outBindIdx;
  std::string _inName, _outName;  // TensorRT 10 I/O tensors are addressed by name

  std::vector<float> proj_xs; // stope a copy in original order
  std::vector<float> proj_ys;

  // explicitly set the invalid point for both inputs and outputs
  std::vector<float> invalid_input =  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> invalid_output = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // timer stuff
  std::vector<std::chrono::system_clock::time_point> stimes;

  void tic()
  {
    stimes.push_back(std::chrono::high_resolution_clock::now());
  }

  /**
   * @brief stops the last timer started and outputs \a msg, if given.
   * @return elapsed time in seconds.
   **/
  double toc()
  {
    assert(stimes.begin() != stimes.end());

    std::chrono::system_clock::time_point endtime = std::chrono::high_resolution_clock::now();
    std::chrono::system_clock::time_point starttime = stimes.back();
    stimes.pop_back();

    std::chrono::duration<double> elapsed_seconds = endtime - starttime;

    return elapsed_seconds.count();
  }
};

}  // namespace segmentation
}  // namespace rangenet
