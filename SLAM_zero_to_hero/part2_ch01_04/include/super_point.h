//
// SuperPoint: Self-Supervised Interest Point Detection and Description
// TensorRT C++ Implementation
//
// Original paper: "SuperPoint: Self-Supervised Interest Point Detection and Description"
// Authors: Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich (Magic Leap)
// CVPR 2018
//

#ifndef SUPER_POINT_H_
#define SUPER_POINT_H_

#include <string>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

#include "buffers.h"
#include "read_config.h"

using tensorrt_common::TensorRTUniquePtr;

/**
 * @brief SuperPoint feature detector and descriptor using TensorRT
 *
 * SuperPoint is a self-supervised deep learning model that jointly detects
 * interest points and computes descriptors in a single forward pass.
 *
 * Architecture:
 * - Shared encoder CNN backbone
 * - Detection head: outputs keypoint scores at 1/8 resolution
 * - Descriptor head: outputs 256-D descriptors at 1/8 resolution
 *
 * Output format (259 x N matrix):
 * - Row 0: keypoint scores
 * - Rows 1-2: keypoint (x, y) coordinates
 * - Rows 3-258: 256-dimensional L2-normalized descriptors
 */
class SuperPoint {
public:
    /**
     * @brief Construct SuperPoint with configuration
     * @param super_point_config Configuration parameters (thresholds, paths, etc.)
     */
    explicit SuperPoint(SuperPointConfig super_point_config);

    /**
     * @brief Build or load TensorRT engine
     *
     * First checks for cached .engine file, if not found:
     * 1. Parses ONNX model
     * 2. Optimizes with TensorRT
     * 3. Builds and caches engine
     *
     * @return true if engine built/loaded successfully
     */
    bool build();

    /**
     * @brief Run inference on input image
     *
     * @param image Grayscale input image (CV_8UC1)
     * @param features Output matrix (259 x N) containing:
     *                 - scores, keypoint coordinates, and descriptors
     * @return true if inference successful
     */
    bool infer(const cv::Mat& image, Eigen::Matrix<double, 259, Eigen::Dynamic>& features);

    /**
     * @brief Visualize detected keypoints on image
     * @param image_name Output filename
     * @param image Input image for visualization
     */
    void visualization(const std::string& image_name, const cv::Mat& image);

    /**
     * @brief Serialize TensorRT engine to file
     * Enables faster loading on subsequent runs
     */
    void save_engine();

    /**
     * @brief Load pre-built TensorRT engine from file
     * @return true if engine loaded successfully
     */
    bool deserialize_engine();

private:
    SuperPointConfig super_point_config_;

    // TensorRT tensor dimensions
    nvinfer1::Dims input_dims_{};      // Input: 1 x 1 x H x W
    nvinfer1::Dims semi_dims_{};       // Scores: 1 x H x W
    nvinfer1::Dims desc_dims_{};       // Descriptors: 1 x 256 x H/8 x W/8

    // TensorRT engine and context
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    // Detection results (before feature matrix conversion)
    std::vector<std::vector<int>> keypoints_;
    std::vector<std::vector<double>> descriptors_;

    /**
     * @brief Construct TensorRT network from ONNX
     */
    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                          TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                          TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                          TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;

    /**
     * @brief Preprocess image and copy to TensorRT buffer
     * Normalizes pixel values to [0, 1] range
     */
    bool process_input(const tensorrt_buffer::BufferManager& buffers, const cv::Mat& image);

    /**
     * @brief Process network output to extract keypoints and descriptors
     */
    bool process_output(const tensorrt_buffer::BufferManager& buffers,
                       Eigen::Matrix<double, 259, Eigen::Dynamic>& features);

    /**
     * @brief Remove keypoints too close to image borders
     */
    void remove_borders(std::vector<std::vector<int>>& keypoints,
                       std::vector<float>& scores,
                       int border, int height, int width);

    /**
     * @brief Sort indices by score (descending)
     */
    std::vector<size_t> sort_indexes(std::vector<float>& data);

    /**
     * @brief Keep only top-k keypoints by score
     */
    void top_k_keypoints(std::vector<std::vector<int>>& keypoints,
                        std::vector<float>& scores, int k);

    /**
     * @brief Filter keypoints by score threshold
     */
    void find_high_score_index(std::vector<float>& scores,
                              std::vector<std::vector<int>>& keypoints,
                              int h, int w, double threshold);

    /**
     * @brief Sample descriptors at keypoint locations using bilinear interpolation
     *
     * Since descriptors are computed at 1/8 resolution, we use bilinear
     * interpolation to sample descriptor vectors at subpixel keypoint locations.
     */
    void sample_descriptors(std::vector<std::vector<int>>& keypoints,
                           float* descriptors,
                           std::vector<std::vector<double>>& dest_descriptors,
                           int dim, int h, int w, int s = 8);
};

// Smart pointer type for SuperPoint
typedef std::shared_ptr<SuperPoint> SuperPointPtr;

#endif // SUPER_POINT_H_
