/* Copyright (c) 2019 Xieyuanli Chen, Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */

#include "netTensorRT.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <iterator>

namespace rangenet {
namespace segmentation {

/**
 * @brief      Constructs the object.
 *
 * @param[in]  model_path  The model path for the inference model directory
 *                         containing the "model.trt" file and the cfg
 */
NetTensorRT::NetTensorRT(const std::string& model_path)
    : Net(model_path), _runtime(0), _engine(0), _context(0) {
  // set default verbosity level
  verbosity(_verbose);

  std::cout << "Trying to open model" << std::endl;

  // TensorRT engines are specific to GPU + TensorRT version, so the cache can
  // be redirected to a writable volume with RANGENET_ENGINE_DIR.
  std::string engine_dir = model_path;
  if (const char* d = std::getenv("RANGENET_ENGINE_DIR")) engine_dir = d;
  std::string engine_path = engine_dir + "/model.trt";

  _runtime = createInferRuntime(_gLogger);
  if (!_runtime) throw std::runtime_error("Couldn't create inference runtime.");

  try {
    deserializeEngine(engine_path);
  } catch (std::exception& e) {
    std::cout << "Could not deserialize TensorRT engine (" << e.what() << "). " << std::endl
              << "Generating from scratch... This may take a while..." << std::endl;
    delete _engine;
    _engine = 0;
  }

  if (!_engine) {
    std::string onnx_path = model_path + "/model.onnx";
    generateEngine(onnx_path);
    serializeEngine(engine_path);
  }

  prepareBuffer();

  CUDA_CHECK(cudaStreamCreate(&_cudaStream));
}

NetTensorRT::~NetTensorRT() {
  for (auto& buffer : _deviceBuffers) CUDA_CHECK(cudaFree(buffer));
  for (auto& buffer : _hostBuffers) CUDA_CHECK(cudaFreeHost(buffer));
  CUDA_CHECK(cudaStreamDestroy(_cudaStream));
  delete _context;
  delete _engine;
  delete _runtime;
}

/**
 * @brief      Project a pointcloud into a spherical projection image.projection.
 *
 * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
 *
 * @return     Projected LiDAR scans, with size of (_img_h * _img_w, _img_d)
 */
std::vector<std::vector<float>> NetTensorRT::doProjection(const std::vector<float>& scan, const uint32_t& num_points){
  float fov_up = _fov_up / 180.0 * M_PI;    // field of view up in radians
  float fov_down = _fov_down / 180.0 * M_PI;  // field of view down in radians
  float fov = std::abs(fov_down) + std::abs(fov_up); // get field of view total in radians

  std::vector<float> ranges;
  std::vector<float> xs;
  std::vector<float> ys;
  std::vector<float> zs;
  std::vector<float> intensitys;

  std::vector<float> proj_xs_tmp;
  std::vector<float> proj_ys_tmp;

  for (uint32_t i = 0; i < num_points; i++) {
    float x = scan[4 * i];
    float y = scan[4 * i + 1];
    float z = scan[4 * i + 2];
    float intensity = scan[4 * i + 3];
    float range = std::sqrt(x*x+y*y+z*z);
    ranges.push_back(range);
    xs.push_back(x);
    ys.push_back(y);
    zs.push_back(z);
    intensitys.push_back(intensity);

    // get angles
    float yaw = -std::atan2(y, x);
    float pitch = std::asin(z / range);

    // get projections in image coords
    float proj_x = 0.5 * (yaw / M_PI + 1.0); // in [0.0, 1.0]
    float proj_y = 1.0 - (pitch + std::abs(fov_down)) / fov; // in [0.0, 1.0]

    // scale to image size using angular resolution
    proj_x *= _img_w; // in [0.0, W]
    proj_y *= _img_h; // in [0.0, H]

    // round and clamp for use as index
    proj_x = std::floor(proj_x);
    proj_x = std::min(_img_w - 1.0f, proj_x);
    proj_x = std::max(0.0f, proj_x); // in [0,W-1]
    proj_xs_tmp.push_back(proj_x);

    proj_y = std::floor(proj_y);
    proj_y = std::min(_img_h - 1.0f, proj_y);
    proj_y = std::max(0.0f, proj_y); // in [0,H-1]
    proj_ys_tmp.push_back(proj_y);
  }

  // stope a copy in original order
  proj_xs = proj_xs_tmp;
  proj_ys = proj_ys_tmp;

  // order in decreasing depth
  std::vector<size_t> orders = sort_indexes(ranges);
  std::vector<float> sorted_proj_xs;
  std::vector<float> sorted_proj_ys;
  std::vector<std::vector<float>> inputs;

  for (size_t idx : orders){
    sorted_proj_xs.push_back(proj_xs[idx]);
    sorted_proj_ys.push_back(proj_ys[idx]);
    std::vector<float> input = {ranges[idx], xs[idx], ys[idx], zs[idx], intensitys[idx]};
    inputs.push_back(input);
  }

  // assing to images
  std::vector<std::vector<float>> range_image(_img_w * _img_h);

  // zero initialize
  for (uint32_t i = 0; i < range_image.size(); ++i) {
      range_image[i] = invalid_input;
  }

  for (uint32_t i = 0; i < inputs.size(); ++i) {
    range_image[int(sorted_proj_ys[i] * _img_w + sorted_proj_xs[i])] = inputs[i];
  }

  return range_image;
}

/**
 * @brief      Infer logits from LiDAR scan
 *
 * @param[in]  scan, LiDAR scans; num_points, the number of points in this scan.
 *
 * @return     Semantic estimates with probabilities over all classes (_n_classes, _img_h, _img_w)
 */
std::vector<std::vector<float>> NetTensorRT::infer(const std::vector<float>& scan, const uint32_t& num_points) {
  // check if engine is valid
  if (!_engine) {
    throw std::runtime_error("Invaild engine on inference.");
  }

  // start inference
  if (_verbose) {
    tic();
    std::cout << "Inferring with TensorRT" << std::endl;
    tic();
  }

  // project point clouds into range image
  std::vector<std::vector<float>> projected_data = doProjection(scan, num_points);


  if (_verbose) {
    std::cout << "Time for projection: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  // put in buffer using position
  int channel_offset = _img_h * _img_w;

  bool all_zeros = false;
  std::vector<int> invalid_idxs;

  for (uint32_t pixel_id = 0; pixel_id < projected_data.size(); pixel_id++){
    // check if the pixel is invalid
    all_zeros = std::all_of(projected_data[pixel_id].begin(), projected_data[pixel_id].end(), [](float i) { return i==0.0f; });
    if (all_zeros) {
      invalid_idxs.push_back(pixel_id);
    }
    for (int i = 0; i < _img_d; i++) {
      // normalize the data
      if (!all_zeros) {
        projected_data[pixel_id][i] = (projected_data[pixel_id][i] - this->_img_means[i]) / this->_img_stds[i];
      }

      int buffer_idx = channel_offset * i + pixel_id;
      ((float*)_hostBuffers[_inBindIdx])[buffer_idx] = projected_data[pixel_id][i];
    }
  }

  // clock now
  if (_verbose) {
    std::cout << "Time for preprocessing: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  // execute inference
  CUDA_CHECK(
      cudaMemcpyAsync(_deviceBuffers[_inBindIdx], _hostBuffers[_inBindIdx],
                      getBufferSize(_engine->getTensorShape(_inName.c_str()),
                                    _engine->getTensorDataType(_inName.c_str())),
                      cudaMemcpyHostToDevice, _cudaStream));
  if (_verbose) {
    CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    std::cout << "Time for copy in: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }


  if (!_context->enqueueV3(_cudaStream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed.");
  }

  if (_verbose) {
    CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    std::cout << "Time for inferring: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  CUDA_CHECK(
      cudaMemcpyAsync(_hostBuffers[_outBindIdx], _deviceBuffers[_outBindIdx],
                      getBufferSize(_engine->getTensorShape(_outName.c_str()),
                                    _engine->getTensorDataType(_outName.c_str())),
                      cudaMemcpyDeviceToHost, _cudaStream));
  CUDA_CHECK(cudaStreamSynchronize(_cudaStream));

  if (_verbose) {
    std::cout << "Time for copy back: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  // take the data out
  std::vector<std::vector<float>> range_image(channel_offset);
  for (int pixel_id = 0; pixel_id < channel_offset; pixel_id++){
    for (int i = 0; i < _n_classes; i++) {
      int buffer_idx = channel_offset * i + pixel_id;
      range_image[pixel_id].push_back(((float*)_hostBuffers[_outBindIdx])[buffer_idx]);
    }
  }

  if (_verbose) {
    std::cout << "Time for taking the data out: "
              << toc() * 1000
              << "ms" << std::endl;
    tic();
  }

  // set invalid pixels
  for (int idx : invalid_idxs) {
    range_image[idx] = invalid_output;
  }

  // unprojection, labelling raw point clouds
  std::vector<std::vector<float>> semantic_scan;
  for (uint32_t i = 0 ; i < num_points; i++) {
    semantic_scan.push_back(range_image[proj_ys[i] * _img_w + proj_xs[i]]);
  }

  if (_verbose) {
    std::cout << "Time for unprojection: "
          << toc() * 1000
          << "ms" << std::endl;
    std::cout << "Time for the whole: "
              << toc() * 1000
              << "ms" << std::endl;
  }

  return semantic_scan;
}

/**
 * @brief      Set verbosity level for backend execution
 *
 * @param[in]  verbose  True is max verbosity, False is no verbosity.
 *
 * @return     Exit code.
 */
void NetTensorRT::verbosity(const bool verbose) {
  std::cout << "Setting verbosity to: " << (verbose ? "true" : "false")
            << std::endl;

  // call parent class verbosity
  this->Net::verbosity(verbose);

  // set verbosity for tensorRT logger
  _gLogger.set_verbosity(verbose);
}

/**
 * @brief Get the Buffer Size object
 *
 * @param d dimension
 * @param t data type
 * @return int size of data
 */
size_t NetTensorRT::getBufferSize(Dims d, DataType t) {
  size_t size = 1;
  for (int i = 0; i < d.nbDims; i++) size *= d.d[i];

  switch (t) {
    case DataType::kINT32:
      return size * 4;
    case DataType::kFLOAT:
      return size * 4;
    case DataType::kHALF:
      return size * 2;
    case DataType::kINT8:
      return size * 1;
    default:
      throw std::runtime_error("Data type not handled");
  }
  return 0;
}

/**
 * @brief Deserialize an engine that comes from a previous run (TensorRT 10 API)
 */
void NetTensorRT::deserializeEngine(const std::string& engine_path) {
  std::cout << "Trying to deserialize previously stored: " << engine_path << std::endl;
  std::ifstream file(engine_path.c_str(), std::ios::binary);
  if (!file) throw std::runtime_error("TensorRT engine file not found " + engine_path);
  std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  std::cout << "Successfully read " << blob.size() << " bytes of engine." << std::endl;
  _engine = _runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!_engine) throw std::runtime_error("Device failed to create CUDA engine");
  std::cout << "Successfully deserialized Engine from trt file" << std::endl;
}

/**
 * @brief Serialize an engine that we generated in this run
 */
void NetTensorRT::serializeEngine(const std::string& engine_path) {
  std::cout << "Trying to serialize engine and save to : " << engine_path << " for next run" << std::endl;
  if (!_engine) return;
  IHostMemory* plan = _engine->serialize();
  std::ofstream stream(engine_path.c_str(), std::ofstream::binary);
  if (stream) stream.write(static_cast<char*>(plan->data()), plan->size());
  else std::cerr << "Could not write " << engine_path << " (engine will be rebuilt next run)" << std::endl;
  delete plan;
}

/**
 * @brief Generate an engine from ONNX model (TensorRT 10 API: explicit batch,
 *        IBuilderConfig, buildSerializedNetwork)
 */
void NetTensorRT::generateEngine(const std::string& onnx_path) {
  std::cout << "Trying to generate trt engine from : " << onnx_path << std::endl;

  IBuilder* builder = createInferBuilder(_gLogger);
  INetworkDefinition* network = builder->createNetworkV2(0);  // explicit batch is the only mode in TRT 10
  IBuilderConfig* config = builder->createBuilderConfig();

  // FP32 by default. The 2019 Dockerfile upstream also turned FP16 off; set
  // RANGENET_FP16=1 to try half precision.
  const char* fp16 = std::getenv("RANGENET_FP16");
  if (fp16 && std::string(fp16) == "1") {
    config->setFlag(BuilderFlag::kFP16);
    std::cout << "Building with FP16." << std::endl;
  } else {
    std::cout << "Building with FP32." << std::endl;
  }
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, MAX_WORKSPACE_SIZE);

  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, _gLogger);
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
    for (int i = 0; i < parser->getNbErrors(); ++i)
      std::cerr << parser->getError(i)->desc() << std::endl;
    throw std::runtime_error("ERROR: could not parse input ONNX.");
  }
  std::cout << "Success picking up ONNX model" << std::endl;

  IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
  if (!plan) throw std::runtime_error("ERROR: could not create engine from ONNX.");
  _engine = _runtime->deserializeCudaEngine(plan->data(), plan->size());
  delete plan;
  delete parser;
  delete config;
  delete network;
  delete builder;
  if (!_engine) throw std::runtime_error("ERROR: could not create engine from ONNX.");
  std::cout << "Success creating engine from ONNX model" << std::endl;
}

/**
 * @brief Prepare io buffers for inference with engine (TensorRT 10 named I/O tensors)
 */
void NetTensorRT::prepareBuffer() {
  if (!_engine) throw std::runtime_error("Invalid engine. Please remember to create engine first.");

  _context = _engine->createExecutionContext();
  if (!_context) throw std::runtime_error("Invalid execution context. Can't infer.");

  int n_io = _engine->getNbIOTensors();
  if (n_io != 2) throw std::runtime_error("Invalid number of I/O tensors: " + std::to_string(n_io));

  _deviceBuffers.assign(n_io, nullptr);
  _hostBuffers.assign(n_io, nullptr);

  for (int i = 0; i < n_io; i++) {
    const char* name = _engine->getIOTensorName(i);
    Dims dims = _engine->getTensorShape(name);
    DataType dtype = _engine->getTensorDataType(name);
    size_t bytes = getBufferSize(dims, dtype);
    CUDA_CHECK(cudaMalloc(&_deviceBuffers[i], bytes));
    CUDA_CHECK(cudaMallocHost(&_hostBuffers[i], bytes));
    _context->setTensorAddress(name, _deviceBuffers[i]);

    if (_engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
      _inBindIdx = i;
      _inName = name;
    } else {
      _outBindIdx = i;
      _outName = name;
    }

    std::cout << "Binding: " << i << " (" << name << "), type: " << (int)dtype << " ";
    for (int d = 0; d < dims.nbDims; d++) std::cout << "[Dim " << dims.d[d] << "]";
    std::cout << std::endl;
  }
  std::cout << "Successfully create binding buffer" << std::endl;
}

}  // namespace segmentation
}  // namespace rangenet
