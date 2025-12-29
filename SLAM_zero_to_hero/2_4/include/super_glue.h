//
// SuperGlue: Learning Feature Matching with Graph Neural Networks
// TensorRT C++ Implementation
//
// Original paper: "SuperGlue: Learning Feature Matching with Graph Neural Networks"
// Authors: Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich
// CVPR 2020
//

#ifndef SUPER_GLUE_H_
#define SUPER_GLUE_H_

#include <string>
#include <memory>
#include <vector>
#include <NvInfer.h>
#include <Eigen/Core>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

#include "buffers.h"
#include "read_config.h"

using tensorrt_common::TensorRTUniquePtr;

/**
 * @brief SuperGlue feature matcher using TensorRT
 *
 * SuperGlue is a learned feature matching network that uses graph neural networks
 * and attention mechanisms to match keypoints between image pairs.
 *
 * Architecture:
 * 1. Keypoint Encoder: Combines position and descriptor information
 * 2. Attentional GNN: Self-attention within images, cross-attention between images
 * 3. Optimal Transport: Sinkhorn algorithm for differentiable matching
 *
 * Key innovations:
 * - Learns to reject outliers during matching (not post-hoc RANSAC)
 * - Considers global context, not just local descriptor similarity
 * - Handles partial visibility with "dustbin" for unmatched points
 */
class SuperGlue {
public:
    /**
     * @brief Construct SuperGlue with configuration
     * @param superglue_config Configuration parameters
     */
    explicit SuperGlue(const SuperGlueConfig& superglue_config);

    /**
     * @brief Build or load TensorRT engine
     * @return true if engine built/loaded successfully
     */
    bool build();

    /**
     * @brief Run inference to match features between two images
     *
     * @param features0 Features from image 0 (259 x N0 matrix)
     * @param features1 Features from image 1 (259 x N1 matrix)
     * @param indices0 Output: for each keypoint in image 0, index of match in image 1 (-1 if unmatched)
     * @param indices1 Output: for each keypoint in image 1, index of match in image 0 (-1 if unmatched)
     * @param mscores0 Output: matching confidence scores for image 0 keypoints
     * @param mscores1 Output: matching confidence scores for image 1 keypoints
     * @return true if inference successful
     */
    bool infer(const Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
              const Eigen::Matrix<double, 259, Eigen::Dynamic>& features1,
              Eigen::VectorXi& indices0,
              Eigen::VectorXi& indices1,
              Eigen::VectorXd& mscores0,
              Eigen::VectorXd& mscores1);

    /**
     * @brief High-level matching function returning OpenCV DMatch format
     *
     * @param features0 Features from image 0
     * @param features1 Features from image 1
     * @param matches Output vector of matches
     * @param outlier_rejection If true, apply RANSAC-based outlier rejection
     * @return Number of matches found
     */
    int matching_points(Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
                       Eigen::Matrix<double, 259, Eigen::Dynamic>& features1,
                       std::vector<cv::DMatch>& matches,
                       bool outlier_rejection = false);

    /**
     * @brief Normalize keypoint coordinates for network input
     *
     * Coordinates are normalized to [-1, 1] range relative to image center
     * and scaled by max(width, height) * 0.7
     *
     * @param features Input features with pixel coordinates
     * @param width Image width
     * @param height Image height
     * @return Features with normalized coordinates
     */
    Eigen::Matrix<double, 259, Eigen::Dynamic> normalize_keypoints(
        const Eigen::Matrix<double, 259, Eigen::Dynamic>& features,
        int width, int height);

    /**
     * @brief Serialize TensorRT engine to file
     */
    void save_engine();

    /**
     * @brief Load pre-built TensorRT engine from file
     * @return true if engine loaded successfully
     */
    bool deserialize_engine();

private:
    SuperGlueConfig superglue_config_;

    // Match results (before Eigen vector conversion)
    std::vector<int> indices0_;
    std::vector<int> indices1_;
    std::vector<double> mscores0_;
    std::vector<double> mscores1_;

    // TensorRT tensor dimensions for dynamic shapes
    // Image 0 inputs
    nvinfer1::Dims keypoints_0_dims_{};    // 1 x N0 x 2
    nvinfer1::Dims scores_0_dims_{};       // 1 x N0
    nvinfer1::Dims descriptors_0_dims_{};  // 1 x 256 x N0

    // Image 1 inputs
    nvinfer1::Dims keypoints_1_dims_{};    // 1 x N1 x 2
    nvinfer1::Dims scores_1_dims_{};       // 1 x N1
    nvinfer1::Dims descriptors_1_dims_{};  // 1 x 256 x N1

    // Output scores (assignment matrix)
    nvinfer1::Dims output_scores_dims_{};  // 1 x (N0+1) x (N1+1)

    // TensorRT engine and context
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    /**
     * @brief Construct TensorRT network from ONNX
     *
     * Sets up dynamic shapes for variable number of keypoints:
     * - MIN: 1 keypoint
     * - OPT: 512 keypoints (typical)
     * - MAX: 1024 keypoints
     */
    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                          TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                          TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                          TensorRTUniquePtr<nvonnxparser::IParser>& parser) const;

    /**
     * @brief Copy feature data to TensorRT input buffers
     *
     * Reorganizes feature matrix into separate tensors:
     * - keypoints: N x 2 (x, y coordinates)
     * - scores: N (keypoint scores)
     * - descriptors: 256 x N (descriptor vectors)
     */
    bool process_input(const tensorrt_buffer::BufferManager& buffers,
                      const Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
                      const Eigen::Matrix<double, 259, Eigen::Dynamic>& features1);

    /**
     * @brief Process network output to extract matches
     *
     * The network outputs an assignment matrix (N0+1) x (N1+1) where:
     * - Entry (i, j) is the log-probability of matching keypoint i to j
     * - Extra row/column is the "dustbin" for unmatched points
     *
     * Decoding:
     * 1. Find max in each row (best match for each keypoint in image 0)
     * 2. Find max in each column (best match for each keypoint in image 1)
     * 3. Keep only mutual matches (i->j AND j->i)
     * 4. Apply confidence threshold
     */
    bool process_output(const tensorrt_buffer::BufferManager& buffers,
                       Eigen::VectorXi& indices0,
                       Eigen::VectorXi& indices1,
                       Eigen::VectorXd& mscores0,
                       Eigen::VectorXd& mscores1);
};

// Smart pointer type for SuperGlue
typedef std::shared_ptr<SuperGlue> SuperGluePtr;

#endif // SUPER_GLUE_H_
