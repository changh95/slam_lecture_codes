//
// SuperPoint TensorRT Implementation
//
// This implementation demonstrates the key steps for deploying
// deep learning feature detectors with TensorRT:
// 1. Build/deserialize TensorRT engine from ONNX
// 2. Preprocess images (normalize, copy to GPU)
// 3. Run inference
// 4. Post-process outputs (NMS, thresholding, descriptor sampling)
//

#include "super_point.h"
#include <utility>
#include <memory>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <opencv2/opencv.hpp>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;

// =============================================================================
// Constructor
// =============================================================================

SuperPoint::SuperPoint(SuperPointConfig super_point_config)
    : super_point_config_(std::move(super_point_config))
    , engine_(nullptr) {
    // Set logging level (suppress verbose output)
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

// =============================================================================
// Engine Building / Loading
// =============================================================================

bool SuperPoint::build() {
    // First, try to load a pre-built engine (much faster)
    if (deserialize_engine()) {
        std::cout << "SuperPoint: Loaded cached TensorRT engine." << std::endl;
        return true;
    }

    std::cout << "SuperPoint: Building TensorRT engine from ONNX..." << std::endl;
    std::cout << "  This may take 10-15 minutes on first run." << std::endl;

    // Create TensorRT builder
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return false;
    }

    // Create network with explicit batch dimension
    const auto explicit_batch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch));
    if (!network) {
        std::cerr << "Failed to create network definition" << std::endl;
        return false;
    }

    // Create builder config
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return false;
    }

    // Create ONNX parser
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return false;
    }

    // Set up optimization profile for dynamic input shapes
    // SuperPoint accepts variable-size images
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Failed to create optimization profile" << std::endl;
        return false;
    }

    // Define input shape ranges: (batch, channels, height, width)
    profile->setDimensions(
        super_point_config_.input_tensor_names[0].c_str(),
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 100, 100));
    profile->setDimensions(
        super_point_config_.input_tensor_names[0].c_str(),
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 500, 500));
    profile->setDimensions(
        super_point_config_.input_tensor_names[0].c_str(),
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 1500, 1500));
    config->addOptimizationProfile(profile);

    // Parse ONNX and configure builder
    if (!construct_network(builder, network, config, parser)) {
        std::cerr << "Failed to construct network from ONNX" << std::endl;
        return false;
    }

    // Build serialized network (this is the slow step)
    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        std::cerr << "Failed to create CUDA stream" << std::endl;
        return false;
    }
    config->setProfileStream(*profile_stream);

    TensorRTUniquePtr<nvinfer1::IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        std::cerr << "Failed to build serialized network" << std::endl;
        return false;
    }

    // Create runtime and deserialize engine
    TensorRTUniquePtr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    // Cache the engine for future runs
    save_engine();

    // Verify network structure
    ASSERT(network->getNbInputs() == 1);
    input_dims_ = network->getInput(0)->getDimensions();
    ASSERT(input_dims_.nbDims == 4);

    ASSERT(network->getNbOutputs() == 2);
    semi_dims_ = network->getOutput(0)->getDimensions();  // Scores
    ASSERT(semi_dims_.nbDims == 3);
    desc_dims_ = network->getOutput(1)->getDimensions();  // Descriptors
    ASSERT(desc_dims_.nbDims == 4);

    std::cout << "SuperPoint: TensorRT engine built successfully." << std::endl;
    return true;
}

bool SuperPoint::construct_network(
    TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
    TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TensorRTUniquePtr<nvonnxparser::IParser>& parser) const {

    // Parse ONNX model file
    auto parsed = parser->parseFromFile(
        super_point_config_.onnx_file.c_str(),
        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        std::cerr << "Failed to parse ONNX file: "
                  << super_point_config_.onnx_file << std::endl;
        return false;
    }

    // Set workspace size (memory for intermediate tensors)
    config->setMaxWorkspaceSize(512 * (1 << 20));  // 512 MB

    // Enable FP16 precision (faster with minimal accuracy loss)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // Enable DLA if configured (for NVIDIA Jetson)
    enableDLA(builder.get(), config.get(), super_point_config_.dla_core);

    return true;
}

// =============================================================================
// Engine Serialization / Deserialization
// =============================================================================

void SuperPoint::save_engine() {
    if (super_point_config_.engine_file.empty()) return;
    if (!engine_) return;

    nvinfer1::IHostMemory* data = engine_->serialize();
    std::ofstream file(super_point_config_.engine_file, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file for writing: "
                  << super_point_config_.engine_file << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(data->data()), data->size());
    std::cout << "SuperPoint: Saved TensorRT engine to "
              << super_point_config_.engine_file << std::endl;
}

bool SuperPoint::deserialize_engine() {
    std::ifstream file(super_point_config_.engine_file, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> model_stream(size);
    file.read(model_stream.data(), size);
    file.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) return false;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(model_stream.data(), size));

    return engine_ != nullptr;
}

// =============================================================================
// Inference
// =============================================================================

bool SuperPoint::infer(const cv::Mat& image,
                       Eigen::Matrix<double, 259, Eigen::Dynamic>& features) {
    // Create execution context on first inference
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
    }

    assert(engine_->getNbBindings() == 3);

    // Set input dimensions for this image
    const int input_index = engine_->getBindingIndex(
        super_point_config_.input_tensor_names[0].c_str());
    context_->setBindingDimensions(
        input_index, nvinfer1::Dims4(1, 1, image.rows, image.cols));

    // Create buffer manager with dynamic shapes
    BufferManager buffers(engine_, 0, context_.get());

    // Preprocess and copy input
    ASSERT(super_point_config_.input_tensor_names.size() == 1);
    if (!process_input(buffers, image)) {
        return false;
    }

    // Copy input to GPU
    buffers.copyInputToDevice();

    // Run inference
    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        std::cerr << "Inference execution failed" << std::endl;
        return false;
    }

    // Copy output to CPU
    buffers.copyOutputToHost();

    // Post-process to extract keypoints and descriptors
    if (!process_output(buffers, features)) {
        return false;
    }

    return true;
}

// =============================================================================
// Input Processing
// =============================================================================

bool SuperPoint::process_input(const BufferManager& buffers, const cv::Mat& image) {
    // Update dimension tracking
    input_dims_.d[2] = image.rows;
    input_dims_.d[3] = image.cols;
    semi_dims_.d[1] = image.rows;
    semi_dims_.d[2] = image.cols;
    desc_dims_.d[1] = 256;
    desc_dims_.d[2] = image.rows / 8;
    desc_dims_.d[3] = image.cols / 8;

    // Get host buffer pointer
    auto* host_data = static_cast<float*>(
        buffers.getHostBuffer(super_point_config_.input_tensor_names[0]));

    // Normalize pixel values to [0, 1] and copy to buffer
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            host_data[row * image.cols + col] =
                static_cast<float>(image.at<unsigned char>(row, col)) / 255.0f;
        }
    }

    return true;
}

// =============================================================================
// Output Processing
// =============================================================================

void SuperPoint::find_high_score_index(std::vector<float>& scores,
                                       std::vector<std::vector<int>>& keypoints,
                                       int h, int w, double threshold) {
    std::vector<float> new_scores;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            std::vector<int> location = {static_cast<int>(i / w),
                                         static_cast<int>(i % w)};
            keypoints.emplace_back(location);
            new_scores.push_back(scores[i]);
        }
    }
    scores.swap(new_scores);
}

void SuperPoint::remove_borders(std::vector<std::vector<int>>& keypoints,
                                std::vector<float>& scores,
                                int border, int height, int width) {
    std::vector<std::vector<int>> keypoints_selected;
    std::vector<float> scores_selected;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        bool valid_h = (keypoints[i][0] >= border) &&
                       (keypoints[i][0] < (height - border));
        bool valid_w = (keypoints[i][1] >= border) &&
                       (keypoints[i][1] < (width - border));

        if (valid_h && valid_w) {
            // Swap x,y for OpenCV convention
            keypoints_selected.push_back({keypoints[i][1], keypoints[i][0]});
            scores_selected.push_back(scores[i]);
        }
    }

    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}

std::vector<size_t> SuperPoint::sort_indexes(std::vector<float>& data) {
    std::vector<size_t> indexes(data.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::sort(indexes.begin(), indexes.end(),
              [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void SuperPoint::top_k_keypoints(std::vector<std::vector<int>>& keypoints,
                                 std::vector<float>& scores, int k) {
    if (k < 0 || static_cast<size_t>(k) >= keypoints.size()) return;

    std::vector<std::vector<int>> keypoints_top_k;
    std::vector<float> scores_top_k;
    std::vector<size_t> indexes = sort_indexes(scores);

    for (int i = 0; i < k; ++i) {
        keypoints_top_k.push_back(keypoints[indexes[i]]);
        scores_top_k.push_back(scores[indexes[i]]);
    }

    keypoints.swap(keypoints_top_k);
    scores.swap(scores_top_k);
}

// Normalize keypoints for descriptor sampling
static void normalize_keypoints_for_sampling(
    const std::vector<std::vector<int>>& keypoints,
    std::vector<std::vector<double>>& keypoints_norm,
    int h, int w, int s) {

    for (const auto& keypoint : keypoints) {
        std::vector<double> kp = {
            keypoint[0] - s / 2.0 + 0.5,
            keypoint[1] - s / 2.0 + 0.5
        };
        kp[0] = kp[0] / (w * s - s / 2.0 - 0.5);
        kp[1] = kp[1] / (h * s - s / 2.0 - 0.5);
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        keypoints_norm.push_back(kp);
    }
}

static int clip(int val, int max_val) {
    if (val < 0) return 0;
    return std::min(val, max_val - 1);
}

// Bilinear interpolation for descriptor sampling
static void grid_sample(const float* input,
                        std::vector<std::vector<double>>& grid,
                        std::vector<std::vector<double>>& output,
                        int dim, int h, int w) {
    for (auto& g : grid) {
        double ix = ((g[0] + 1) / 2) * (w - 1);
        double iy = ((g[1] + 1) / 2) * (h - 1);

        int ix_nw = clip(static_cast<int>(std::floor(ix)), w);
        int iy_nw = clip(static_cast<int>(std::floor(iy)), h);
        int ix_ne = clip(ix_nw + 1, w);
        int iy_ne = clip(iy_nw, h);
        int ix_sw = clip(ix_nw, w);
        int iy_sw = clip(iy_nw + 1, h);
        int ix_se = clip(ix_nw + 1, w);
        int iy_se = clip(iy_nw + 1, h);

        double nw = (ix_se - ix) * (iy_se - iy);
        double ne = (ix - ix_sw) * (iy_sw - iy);
        double sw = (ix_ne - ix) * (iy - iy_ne);
        double se = (ix - ix_nw) * (iy - iy_nw);

        std::vector<double> descriptor;
        for (int i = 0; i < dim; ++i) {
            float nw_val = input[i * h * w + iy_nw * w + ix_nw];
            float ne_val = input[i * h * w + iy_ne * w + ix_ne];
            float sw_val = input[i * h * w + iy_sw * w + ix_sw];
            float se_val = input[i * h * w + iy_se * w + ix_se];
            descriptor.push_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
        }
        output.push_back(descriptor);
    }
}

// L2 normalization
template<typename Iter_T>
static double vector_normalize(Iter_T first, Iter_T last) {
    return std::sqrt(std::inner_product(first, last, first, 0.0));
}

static void normalize_descriptors(std::vector<std::vector<double>>& descriptors) {
    for (auto& desc : descriptors) {
        double norm_inv = 1.0 / vector_normalize(desc.begin(), desc.end());
        std::transform(desc.begin(), desc.end(), desc.begin(),
                      [norm_inv](double v) { return v * norm_inv; });
    }
}

void SuperPoint::sample_descriptors(std::vector<std::vector<int>>& keypoints,
                                    float* descriptors,
                                    std::vector<std::vector<double>>& dest_descriptors,
                                    int dim, int h, int w, int s) {
    std::vector<std::vector<double>> keypoints_norm;
    normalize_keypoints_for_sampling(keypoints, keypoints_norm, h, w, s);
    grid_sample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
    normalize_descriptors(dest_descriptors);
}

bool SuperPoint::process_output(const BufferManager& buffers,
                                Eigen::Matrix<double, 259, Eigen::Dynamic>& features) {
    keypoints_.clear();
    descriptors_.clear();

    // Get output buffers
    auto* output_score = static_cast<float*>(
        buffers.getHostBuffer(super_point_config_.output_tensor_names[0]));
    auto* output_desc = static_cast<float*>(
        buffers.getHostBuffer(super_point_config_.output_tensor_names[1]));

    int semi_h = semi_dims_.d[1];
    int semi_w = semi_dims_.d[2];

    // Convert to vector for processing
    std::vector<float> scores_vec(output_score, output_score + semi_h * semi_w);

    // Filter by score threshold
    find_high_score_index(scores_vec, keypoints_, semi_h, semi_w,
                         super_point_config_.keypoint_threshold);

    // Remove border keypoints
    remove_borders(keypoints_, scores_vec, super_point_config_.remove_borders,
                  semi_h, semi_w);

    // Keep top-k keypoints
    top_k_keypoints(keypoints_, scores_vec, super_point_config_.max_keypoints);

    // Prepare output matrix
    features.resize(259, scores_vec.size());

    // Sample descriptors at keypoint locations
    int desc_dim = desc_dims_.d[1];
    int desc_h = desc_dims_.d[2];
    int desc_w = desc_dims_.d[3];
    sample_descriptors(keypoints_, output_desc, descriptors_,
                      desc_dim, desc_h, desc_w);

    // Fill output matrix
    // Row 0: scores
    for (size_t i = 0; i < scores_vec.size(); i++) {
        features(0, i) = scores_vec[i];
    }

    // Rows 1-2: keypoint coordinates (x, y)
    for (size_t i = 0; i < keypoints_.size(); ++i) {
        features(1, i) = keypoints_[i][0];  // x
        features(2, i) = keypoints_[i][1];  // y
    }

    // Rows 3-258: 256-dimensional descriptors
    for (size_t i = 0; i < descriptors_.size(); ++i) {
        for (int d = 0; d < 256; ++d) {
            features(3 + d, i) = descriptors_[i][d];
        }
    }

    return true;
}

// =============================================================================
// Visualization
// =============================================================================

void SuperPoint::visualization(const std::string& image_name, const cv::Mat& image) {
    cv::Mat image_display;
    if (image.channels() == 1) {
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    } else {
        image_display = image.clone();
    }

    for (const auto& keypoint : keypoints_) {
        cv::circle(image_display, cv::Point(keypoint[0], keypoint[1]),
                  2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }

    cv::imwrite(image_name + "_keypoints.jpg", image_display);
}
