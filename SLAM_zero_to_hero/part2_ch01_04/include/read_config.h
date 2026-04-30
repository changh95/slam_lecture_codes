//
// Configuration Reader for SuperPoint and SuperGlue
// Parses YAML configuration files
//

#ifndef READ_CONFIG_H_
#define READ_CONFIG_H_

#include "utils.h"
#include <iostream>
#include <yaml-cpp/yaml.h>

/**
 * @brief SuperPoint configuration parameters
 */
struct SuperPointConfig {
    // Detection parameters
    int max_keypoints{};           // Maximum number of keypoints to detect (-1 for unlimited)
    double keypoint_threshold{};   // Score threshold for keypoint detection
    int remove_borders{};          // Border margin to exclude keypoints (pixels)

    // TensorRT parameters
    int dla_core{};                // DLA core to use (-1 for GPU only)

    // Tensor names (must match ONNX model)
    std::vector<std::string> input_tensor_names;   // Usually: ["input"]
    std::vector<std::string> output_tensor_names;  // Usually: ["scores", "descriptors"]

    // Model file paths
    std::string onnx_file;         // Path to ONNX model
    std::string engine_file;       // Path to cached TensorRT engine
};

/**
 * @brief SuperGlue configuration parameters
 */
struct SuperGlueConfig {
    // Image dimensions (for keypoint normalization)
    int image_width{};
    int image_height{};

    // TensorRT parameters
    int dla_core{};  // DLA core to use (-1 for GPU only)

    // Tensor names (must match ONNX model)
    std::vector<std::string> input_tensor_names;
    // Usually: ["keypoints_0", "scores_0", "descriptors_0",
    //           "keypoints_1", "scores_1", "descriptors_1"]

    std::vector<std::string> output_tensor_names;  // Usually: ["scores"]

    // Model file paths
    std::string onnx_file;
    std::string engine_file;
};

/**
 * @brief Main configuration container
 *
 * Loads configuration from YAML file with the following structure:
 *
 * superpoint:
 *   max_keypoints: 300
 *   keypoint_threshold: 0.004
 *   remove_borders: 4
 *   input_tensor_names: ["input"]
 *   output_tensor_names: ["scores", "descriptors"]
 *   onnx_file: "superpoint_v1.onnx"
 *   engine_file: "superpoint_v1.engine"
 *   dla_core: -1
 *
 * superglue:
 *   image_width: 640
 *   image_height: 480
 *   input_tensor_names: [...]
 *   output_tensor_names: ["scores"]
 *   onnx_file: "superglue_indoor.onnx"
 *   engine_file: "superglue_indoor.engine"
 *   dla_core: -1
 */
struct Configs {
    std::string model_dir;

    SuperPointConfig superpoint_config;
    SuperGlueConfig superglue_config;

    /**
     * @brief Load configuration from YAML file
     * @param config_file Path to YAML configuration file
     * @param model_dir Directory containing ONNX and engine files
     */
    Configs(const std::string& config_file, const std::string& model_dir) {
        std::cout << "Loading config from: " << config_file << std::endl;

        if (!FileExists(config_file)) {
            std::cerr << "Error: Config file " << config_file << " not found." << std::endl;
            return;
        }

        YAML::Node file_node = YAML::LoadFile(config_file);

        // ===== Parse SuperPoint configuration =====
        YAML::Node superpoint_node = file_node["superpoint"];

        superpoint_config.max_keypoints = superpoint_node["max_keypoints"].as<int>();
        superpoint_config.keypoint_threshold = superpoint_node["keypoint_threshold"].as<double>();
        superpoint_config.remove_borders = superpoint_node["remove_borders"].as<int>();
        superpoint_config.dla_core = superpoint_node["dla_core"].as<int>();

        // Parse input tensor names
        YAML::Node sp_input_names = superpoint_node["input_tensor_names"];
        for (size_t i = 0; i < sp_input_names.size(); i++) {
            superpoint_config.input_tensor_names.push_back(sp_input_names[i].as<std::string>());
        }

        // Parse output tensor names
        YAML::Node sp_output_names = superpoint_node["output_tensor_names"];
        for (size_t i = 0; i < sp_output_names.size(); i++) {
            superpoint_config.output_tensor_names.push_back(sp_output_names[i].as<std::string>());
        }

        // Resolve model file paths
        std::string sp_onnx = superpoint_node["onnx_file"].as<std::string>();
        std::string sp_engine = superpoint_node["engine_file"].as<std::string>();
        superpoint_config.onnx_file = ConcatenateFolderAndFileName(model_dir, sp_onnx);
        superpoint_config.engine_file = ConcatenateFolderAndFileName(model_dir, sp_engine);

        // ===== Parse SuperGlue configuration =====
        YAML::Node superglue_node = file_node["superglue"];

        superglue_config.image_width = superglue_node["image_width"].as<int>();
        superglue_config.image_height = superglue_node["image_height"].as<int>();
        superglue_config.dla_core = superglue_node["dla_core"].as<int>();

        // Parse input tensor names
        YAML::Node sg_input_names = superglue_node["input_tensor_names"];
        for (size_t i = 0; i < sg_input_names.size(); i++) {
            superglue_config.input_tensor_names.push_back(sg_input_names[i].as<std::string>());
        }

        // Parse output tensor names
        YAML::Node sg_output_names = superglue_node["output_tensor_names"];
        for (size_t i = 0; i < sg_output_names.size(); i++) {
            superglue_config.output_tensor_names.push_back(sg_output_names[i].as<std::string>());
        }

        // Resolve model file paths
        std::string sg_onnx = superglue_node["onnx_file"].as<std::string>();
        std::string sg_engine = superglue_node["engine_file"].as<std::string>();
        superglue_config.onnx_file = ConcatenateFolderAndFileName(model_dir, sg_onnx);
        superglue_config.engine_file = ConcatenateFolderAndFileName(model_dir, sg_engine);

        std::cout << "Configuration loaded successfully." << std::endl;
        std::cout << "  SuperPoint ONNX: " << superpoint_config.onnx_file << std::endl;
        std::cout << "  SuperGlue ONNX: " << superglue_config.onnx_file << std::endl;
    }
};

#endif // READ_CONFIG_H_
