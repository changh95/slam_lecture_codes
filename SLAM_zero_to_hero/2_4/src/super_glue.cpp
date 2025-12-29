//
// SuperGlue TensorRT Implementation
//
// SuperGlue uses graph neural networks with attention mechanisms
// to match features between two images. Key components:
// 1. Keypoint encoder with positional encoding
// 2. Alternating self-attention and cross-attention layers
// 3. Optimal transport (Sinkhorn) for matching assignment
//

#include "super_glue.h"
#include <cfloat>
#include <utility>
#include <fstream>
#include <unordered_map>
#include <opencv2/opencv.hpp>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;

// =============================================================================
// Constructor
// =============================================================================

SuperGlue::SuperGlue(const SuperGlueConfig& superglue_config)
    : superglue_config_(superglue_config)
    , engine_(nullptr) {
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

// =============================================================================
// Engine Building / Loading
// =============================================================================

bool SuperGlue::build() {
    // Try to load cached engine first
    if (deserialize_engine()) {
        std::cout << "SuperGlue: Loaded cached TensorRT engine." << std::endl;
        return true;
    }

    std::cout << "SuperGlue: Building TensorRT engine from ONNX..." << std::endl;
    std::cout << "  This may take 15-20 minutes on first run." << std::endl;

    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) return false;

    const auto explicit_batch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch));
    if (!network) return false;

    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) return false;

    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) return false;

    // Create optimization profile for dynamic number of keypoints
    auto profile = builder->createOptimizationProfile();
    if (!profile) return false;

    // Configure dynamic shapes for keypoints from image 0
    // keypoints_0: 1 x N x 2
    profile->setDimensions(
        superglue_config_.input_tensor_names[0].c_str(),
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2));
    profile->setDimensions(
        superglue_config_.input_tensor_names[0].c_str(),
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 2));
    profile->setDimensions(
        superglue_config_.input_tensor_names[0].c_str(),
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 1024, 2));

    // scores_0: 1 x N
    profile->setDimensions(
        superglue_config_.input_tensor_names[1].c_str(),
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions(
        superglue_config_.input_tensor_names[1].c_str(),
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 512));
    profile->setDimensions(
        superglue_config_.input_tensor_names[1].c_str(),
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 1024));

    // descriptors_0: 1 x 256 x N
    profile->setDimensions(
        superglue_config_.input_tensor_names[2].c_str(),
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 256, 1));
    profile->setDimensions(
        superglue_config_.input_tensor_names[2].c_str(),
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 256, 512));
    profile->setDimensions(
        superglue_config_.input_tensor_names[2].c_str(),
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 256, 1024));

    // Configure dynamic shapes for keypoints from image 1
    // keypoints_1: 1 x M x 2
    profile->setDimensions(
        superglue_config_.input_tensor_names[3].c_str(),
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2));
    profile->setDimensions(
        superglue_config_.input_tensor_names[3].c_str(),
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 2));
    profile->setDimensions(
        superglue_config_.input_tensor_names[3].c_str(),
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 1024, 2));

    // scores_1: 1 x M
    profile->setDimensions(
        superglue_config_.input_tensor_names[4].c_str(),
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions(
        superglue_config_.input_tensor_names[4].c_str(),
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 512));
    profile->setDimensions(
        superglue_config_.input_tensor_names[4].c_str(),
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 1024));

    // descriptors_1: 1 x 256 x M
    profile->setDimensions(
        superglue_config_.input_tensor_names[5].c_str(),
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 256, 1));
    profile->setDimensions(
        superglue_config_.input_tensor_names[5].c_str(),
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 256, 512));
    profile->setDimensions(
        superglue_config_.input_tensor_names[5].c_str(),
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 256, 1024));

    config->addOptimizationProfile(profile);

    if (!construct_network(builder, network, config, parser)) {
        return false;
    }

    auto profile_stream = makeCudaStream();
    if (!profile_stream) return false;
    config->setProfileStream(*profile_stream);

    TensorRTUniquePtr<nvinfer1::IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    if (!plan) return false;

    TensorRTUniquePtr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) return false;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) return false;

    save_engine();

    // Verify network structure
    ASSERT(network->getNbInputs() == 6);
    keypoints_0_dims_ = network->getInput(0)->getDimensions();
    scores_0_dims_ = network->getInput(1)->getDimensions();
    descriptors_0_dims_ = network->getInput(2)->getDimensions();
    keypoints_1_dims_ = network->getInput(3)->getDimensions();
    scores_1_dims_ = network->getInput(4)->getDimensions();
    descriptors_1_dims_ = network->getInput(5)->getDimensions();

    std::cout << "SuperGlue: TensorRT engine built successfully." << std::endl;
    return true;
}

bool SuperGlue::construct_network(
    TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
    TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TensorRTUniquePtr<nvonnxparser::IParser>& parser) const {

    auto parsed = parser->parseFromFile(
        superglue_config_.onnx_file.c_str(),
        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        std::cerr << "Failed to parse ONNX file: "
                  << superglue_config_.onnx_file << std::endl;
        return false;
    }

    config->setMaxWorkspaceSize(512 * (1 << 20));  // 512 MB
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    enableDLA(builder.get(), config.get(), superglue_config_.dla_core);

    return true;
}

// =============================================================================
// Engine Serialization / Deserialization
// =============================================================================

void SuperGlue::save_engine() {
    if (superglue_config_.engine_file.empty()) return;
    if (!engine_) return;

    nvinfer1::IHostMemory* data = engine_->serialize();
    std::ofstream file(superglue_config_.engine_file, std::ios::binary);
    if (!file) return;

    file.write(reinterpret_cast<const char*>(data->data()), data->size());
    std::cout << "SuperGlue: Saved TensorRT engine to "
              << superglue_config_.engine_file << std::endl;
}

bool SuperGlue::deserialize_engine() {
    std::ifstream file(superglue_config_.engine_file, std::ios::binary);
    if (!file.is_open()) return false;

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

bool SuperGlue::infer(const Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
                      const Eigen::Matrix<double, 259, Eigen::Dynamic>& features1,
                      Eigen::VectorXi& indices0,
                      Eigen::VectorXi& indices1,
                      Eigen::VectorXd& mscores0,
                      Eigen::VectorXd& mscores1) {

    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        if (!context_) return false;
    }

    assert(engine_->getNbBindings() == 7);

    // Get binding indices
    const int kp0_idx = engine_->getBindingIndex(superglue_config_.input_tensor_names[0].c_str());
    const int sc0_idx = engine_->getBindingIndex(superglue_config_.input_tensor_names[1].c_str());
    const int desc0_idx = engine_->getBindingIndex(superglue_config_.input_tensor_names[2].c_str());
    const int kp1_idx = engine_->getBindingIndex(superglue_config_.input_tensor_names[3].c_str());
    const int sc1_idx = engine_->getBindingIndex(superglue_config_.input_tensor_names[4].c_str());
    const int desc1_idx = engine_->getBindingIndex(superglue_config_.input_tensor_names[5].c_str());
    const int out_idx = engine_->getBindingIndex(superglue_config_.output_tensor_names[0].c_str());

    // Set dynamic dimensions
    context_->setBindingDimensions(kp0_idx, nvinfer1::Dims3(1, features0.cols(), 2));
    context_->setBindingDimensions(sc0_idx, nvinfer1::Dims2(1, features0.cols()));
    context_->setBindingDimensions(desc0_idx, nvinfer1::Dims3(1, 256, features0.cols()));
    context_->setBindingDimensions(kp1_idx, nvinfer1::Dims3(1, features1.cols(), 2));
    context_->setBindingDimensions(sc1_idx, nvinfer1::Dims2(1, features1.cols()));
    context_->setBindingDimensions(desc1_idx, nvinfer1::Dims3(1, 256, features1.cols()));

    // Get actual dimensions
    keypoints_0_dims_ = context_->getBindingDimensions(kp0_idx);
    scores_0_dims_ = context_->getBindingDimensions(sc0_idx);
    descriptors_0_dims_ = context_->getBindingDimensions(desc0_idx);
    keypoints_1_dims_ = context_->getBindingDimensions(kp1_idx);
    scores_1_dims_ = context_->getBindingDimensions(sc1_idx);
    descriptors_1_dims_ = context_->getBindingDimensions(desc1_idx);
    output_scores_dims_ = context_->getBindingDimensions(out_idx);

    BufferManager buffers(engine_, 0, context_.get());

    ASSERT(superglue_config_.input_tensor_names.size() == 6);
    if (!process_input(buffers, features0, features1)) {
        return false;
    }

    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) return false;

    buffers.copyOutputToHost();

    if (!process_output(buffers, indices0, indices1, mscores0, mscores1)) {
        return false;
    }

    return true;
}

// =============================================================================
// Input Processing
// =============================================================================

bool SuperGlue::process_input(const BufferManager& buffers,
                              const Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
                              const Eigen::Matrix<double, 259, Eigen::Dynamic>& features1) {

    // Get buffer pointers
    auto* kp0_buf = static_cast<float*>(
        buffers.getHostBuffer(superglue_config_.input_tensor_names[0]));
    auto* sc0_buf = static_cast<float*>(
        buffers.getHostBuffer(superglue_config_.input_tensor_names[1]));
    auto* desc0_buf = static_cast<float*>(
        buffers.getHostBuffer(superglue_config_.input_tensor_names[2]));
    auto* kp1_buf = static_cast<float*>(
        buffers.getHostBuffer(superglue_config_.input_tensor_names[3]));
    auto* sc1_buf = static_cast<float*>(
        buffers.getHostBuffer(superglue_config_.input_tensor_names[4]));
    auto* desc1_buf = static_cast<float*>(
        buffers.getHostBuffer(superglue_config_.input_tensor_names[5]));

    // Copy features0: scores (row 0)
    for (Eigen::Index i = 0; i < features0.cols(); ++i) {
        sc0_buf[i] = static_cast<float>(features0(0, i));
    }

    // Copy features0: keypoints (rows 1-2) -> N x 2
    for (Eigen::Index i = 0; i < features0.cols(); ++i) {
        kp0_buf[i * 2 + 0] = static_cast<float>(features0(1, i));  // x
        kp0_buf[i * 2 + 1] = static_cast<float>(features0(2, i));  // y
    }

    // Copy features0: descriptors (rows 3-258) -> 256 x N
    for (int d = 0; d < 256; ++d) {
        for (Eigen::Index i = 0; i < features0.cols(); ++i) {
            desc0_buf[d * features0.cols() + i] = static_cast<float>(features0(3 + d, i));
        }
    }

    // Copy features1: scores
    for (Eigen::Index i = 0; i < features1.cols(); ++i) {
        sc1_buf[i] = static_cast<float>(features1(0, i));
    }

    // Copy features1: keypoints
    for (Eigen::Index i = 0; i < features1.cols(); ++i) {
        kp1_buf[i * 2 + 0] = static_cast<float>(features1(1, i));
        kp1_buf[i * 2 + 1] = static_cast<float>(features1(2, i));
    }

    // Copy features1: descriptors
    for (int d = 0; d < 256; ++d) {
        for (Eigen::Index i = 0; i < features1.cols(); ++i) {
            desc1_buf[d * features1.cols() + i] = static_cast<float>(features1(3 + d, i));
        }
    }

    return true;
}

// =============================================================================
// Output Processing (Match Decoding)
// =============================================================================

// Find max value and index along a dimension
static void max_matrix(const float* data, int* indices, float* values,
                       int h, int w, int dim) {
    if (dim == 2) {
        // Max along columns (for each row)
        for (int i = 0; i < h - 1; ++i) {
            float max_val = -FLT_MAX;
            int max_idx = 0;
            for (int j = 0; j < w - 1; ++j) {
                if (data[i * w + j] > max_val) {
                    max_val = data[i * w + j];
                    max_idx = j;
                }
            }
            values[i] = max_val;
            indices[i] = max_idx;
        }
    } else if (dim == 1) {
        // Max along rows (for each column)
        for (int j = 0; j < w - 1; ++j) {
            float max_val = -FLT_MAX;
            int max_idx = 0;
            for (int i = 0; i < h - 1; ++i) {
                if (data[i * w + j] > max_val) {
                    max_val = data[i * w + j];
                    max_idx = i;
                }
            }
            values[j] = max_val;
            indices[j] = max_idx;
        }
    }
}

// Check for mutual matches
static void equal_gather(const int* indices0, const int* indices1,
                         int* mutual, int size) {
    for (int i = 0; i < size; ++i) {
        mutual[i] = (indices0[indices1[i]] == i) ? 1 : 0;
    }
}

static void where_exp(const int* flag, float* data,
                      std::vector<double>& mscores, int size) {
    for (int i = 0; i < size; ++i) {
        mscores.push_back(flag[i] ? std::exp(data[i]) : 0.0);
    }
}

static void where_gather(const int* flag, int* indices,
                         std::vector<double>& mscores0,
                         std::vector<double>& mscores1, int size) {
    for (int i = 0; i < size; ++i) {
        mscores1.push_back(flag[i] ? mscores0[indices[i]] : 0.0);
    }
}

static void and_threshold(const int* mutual, int* valid,
                          const std::vector<double>& mscores, double threshold) {
    for (size_t i = 0; i < mscores.size(); ++i) {
        valid[i] = (mutual[i] && mscores[i] > threshold) ? 1 : 0;
    }
}

static void and_gather(const int* mutual1, const int* valid0, const int* indices1,
                       int* valid1, int size) {
    for (int i = 0; i < size; ++i) {
        valid1[i] = (mutual1[i] && valid0[indices1[i]]) ? 1 : 0;
    }
}

static void where_negative_one(const int* flag, const int* data,
                               std::vector<int>& indices, int size) {
    for (int i = 0; i < size; ++i) {
        indices.push_back(flag[i] ? data[i] : -1);
    }
}

/**
 * @brief Decode assignment matrix to matches
 *
 * The network outputs a (N0+1) x (N1+1) assignment matrix where:
 * - Entry (i, j) is the log-probability of matching keypoint i to j
 * - Row N0 and column N1 are "dustbin" for unmatched points
 *
 * Decoding process:
 * 1. Find best match for each keypoint in image 0 (argmax along columns)
 * 2. Find best match for each keypoint in image 1 (argmax along rows)
 * 3. Keep only mutual matches (i -> j AND j -> i)
 * 4. Apply confidence threshold
 */
static void decode(float* scores, int h, int w,
                   std::vector<int>& indices0, std::vector<int>& indices1,
                   std::vector<double>& mscores0, std::vector<double>& mscores1) {

    std::vector<int> max_indices0(h - 1);
    std::vector<int> max_indices1(w - 1);
    std::vector<float> max_values0(h - 1);
    std::vector<float> max_values1(w - 1);

    max_matrix(scores, max_indices0.data(), max_values0.data(), h, w, 2);
    max_matrix(scores, max_indices1.data(), max_values1.data(), h, w, 1);

    std::vector<int> mutual0(h - 1);
    std::vector<int> mutual1(w - 1);
    equal_gather(max_indices1.data(), max_indices0.data(), mutual0.data(), h - 1);
    equal_gather(max_indices0.data(), max_indices1.data(), mutual1.data(), w - 1);

    where_exp(mutual0.data(), max_values0.data(), mscores0, h - 1);
    where_gather(mutual1.data(), max_indices1.data(), mscores0, mscores1, w - 1);

    std::vector<int> valid0(h - 1);
    std::vector<int> valid1(w - 1);
    and_threshold(mutual0.data(), valid0.data(), mscores0, 0.2);
    and_gather(mutual1.data(), valid0.data(), max_indices1.data(), valid1.data(), w - 1);

    where_negative_one(valid0.data(), max_indices0.data(), indices0, h - 1);
    where_negative_one(valid1.data(), max_indices1.data(), indices1, w - 1);
}

bool SuperGlue::process_output(const BufferManager& buffers,
                               Eigen::VectorXi& indices0,
                               Eigen::VectorXi& indices1,
                               Eigen::VectorXd& mscores0,
                               Eigen::VectorXd& mscores1) {
    indices0_.clear();
    indices1_.clear();
    mscores0_.clear();
    mscores1_.clear();

    auto* output_score = static_cast<float*>(
        buffers.getHostBuffer(superglue_config_.output_tensor_names[0]));

    int h = output_scores_dims_.d[1];
    int w = output_scores_dims_.d[2];

    decode(output_score, h, w, indices0_, indices1_, mscores0_, mscores1_);

    // Convert to Eigen vectors
    indices0.resize(indices0_.size());
    indices1.resize(indices1_.size());
    mscores0.resize(mscores0_.size());
    mscores1.resize(mscores1_.size());

    for (size_t i = 0; i < indices0_.size(); ++i) {
        indices0(i) = indices0_[i];
    }
    for (size_t i = 0; i < indices1_.size(); ++i) {
        indices1(i) = indices1_[i];
    }
    for (size_t i = 0; i < mscores0_.size(); ++i) {
        mscores0(i) = mscores0_[i];
    }
    for (size_t i = 0; i < mscores1_.size(); ++i) {
        mscores1(i) = mscores1_[i];
    }

    return true;
}

// =============================================================================
// High-Level Matching Interface
// =============================================================================

Eigen::Matrix<double, 259, Eigen::Dynamic> SuperGlue::normalize_keypoints(
    const Eigen::Matrix<double, 259, Eigen::Dynamic>& features,
    int width, int height) {

    Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features;
    norm_features.resize(259, features.cols());
    norm_features = features;

    // Normalize coordinates relative to image center
    // Scale by max(width, height) * 0.7
    double scale = std::max(width, height) * 0.7;

    for (Eigen::Index i = 0; i < features.cols(); ++i) {
        norm_features(1, i) = (features(1, i) - width / 2.0) / scale;   // x
        norm_features(2, i) = (features(2, i) - height / 2.0) / scale;  // y
    }

    return norm_features;
}

int SuperGlue::matching_points(Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
                               Eigen::Matrix<double, 259, Eigen::Dynamic>& features1,
                               std::vector<cv::DMatch>& matches,
                               bool outlier_rejection) {
    matches.clear();

    // Normalize keypoint coordinates for network input
    auto norm_features0 = normalize_keypoints(features0,
        superglue_config_.image_width, superglue_config_.image_height);
    auto norm_features1 = normalize_keypoints(features1,
        superglue_config_.image_width, superglue_config_.image_height);

    // Run inference
    Eigen::VectorXi indices0, indices1;
    Eigen::VectorXd mscores0, mscores1;
    infer(norm_features0, norm_features1, indices0, indices1, mscores0, mscores1);

    // Convert to OpenCV DMatch format
    std::vector<cv::Point2f> points0, points1;

    for (Eigen::Index i = 0; i < indices0.size(); ++i) {
        if (indices0(i) >= 0 && indices0(i) < indices1.size() &&
            indices1(indices0(i)) == static_cast<int>(i)) {

            // Compute match distance as 1 - average confidence
            double dist = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
            matches.emplace_back(static_cast<int>(i), indices0[i], static_cast<float>(dist));

            points0.emplace_back(features0(1, i), features0(2, i));
            points1.emplace_back(features1(1, indices0(i)), features1(2, indices0(i)));
        }
    }

    // Optional: RANSAC-based outlier rejection
    if (outlier_rejection && matches.size() >= 8) {
        std::vector<uchar> inliers;
        cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3.0, 0.99, inliers);

        std::vector<cv::DMatch> inlier_matches;
        for (size_t i = 0; i < matches.size(); ++i) {
            if (inliers[i]) {
                inlier_matches.push_back(matches[i]);
            }
        }
        matches = std::move(inlier_matches);
    }

    return static_cast<int>(matches.size());
}
