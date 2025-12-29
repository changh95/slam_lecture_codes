/**
 * ONNX Runtime Inference for Global Feature Extraction
 *
 * This example demonstrates:
 * 1. Loading a pre-trained model exported to ONNX format
 * 2. Image preprocessing for the model
 * 3. Running inference with ONNX Runtime
 * 4. Using the extracted global descriptors for image retrieval
 *
 * This enables deploying PyTorch/TensorFlow models in C++ without
 * the full framework dependencies.
 *
 * Prerequisites:
 * - Export your model using examples/export_onnx.py
 * - Install ONNX Runtime from https://onnxruntime.ai/
 */

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef ONNXRUNTIME_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

/**
 * Preprocess image for CNN input
 */
std::vector<float> preprocessImage(const cv::Mat& image,
                                   int target_height, int target_width) {
    cv::Mat resized, float_img, normalized;

    // Resize
    cv::resize(image, resized, cv::Size(target_width, target_height));

    // Convert to float
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // ImageNet normalization
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    // Mean and std for ImageNet
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / std[c];
    }

    // Convert to CHW format (PyTorch convention)
    std::vector<float> tensor_data;
    tensor_data.reserve(3 * target_height * target_width);

    // Note: OpenCV uses BGR, but our model expects RGB
    // channels[0] = B, channels[1] = G, channels[2] = R
    for (int c = 2; c >= 0; --c) {  // RGB order
        for (int h = 0; h < target_height; ++h) {
            for (int w = 0; w < target_width; ++w) {
                tensor_data.push_back(channels[c].at<float>(h, w));
            }
        }
    }

    return tensor_data;
}

/**
 * Compute cosine similarity between two vectors
 */
float cosineSimilarity(const std::vector<float>& a,
                       const std::vector<float>& b) {
    if (a.size() != b.size()) return 0.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

/**
 * L2 normalize a vector
 */
void l2Normalize(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float v : vec) norm += v * v;
    norm = std::sqrt(norm);

    if (norm > 0.0f) {
        for (float& v : vec) v /= norm;
    }
}

#ifdef ONNXRUNTIME_AVAILABLE
/**
 * ONNX Runtime inference class
 */
class ONNXFeatureExtractor {
public:
    ONNXFeatureExtractor(const std::string& model_path,
                         int input_height = 480,
                         int input_width = 640)
        : env_(ORT_LOGGING_LEVEL_WARNING, "FeatureExtractor"),
          input_height_(input_height),
          input_width_(input_width) {

        // Session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create session
        session_ = std::make_unique<Ort::Session>(
            env_, model_path.c_str(), session_options);

        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;

        // Input
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name.get();

        auto input_shape = session_->GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo().GetShape();

        std::cout << "Model input: " << input_name_ << " [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Output
        auto output_name = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name.get();

        auto output_shape = session_->GetOutputTypeInfo(0)
            .GetTensorTypeAndShapeInfo().GetShape();
        output_dim_ = output_shape[1];

        std::cout << "Model output: " << output_name_ << " [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::vector<float> extractDescriptor(const cv::Mat& image) {
        // Preprocess
        auto input_tensor_data = preprocessImage(
            image, input_height_, input_width_);

        // Create input tensor
        std::vector<int64_t> input_shape = {
            1, 3, input_height_, input_width_};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_data.data(), input_tensor_data.size(),
            input_shape.data(), input_shape.size());

        // Run inference
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> descriptor(output_data, output_data + output_dim_);

        // L2 normalize
        l2Normalize(descriptor);

        return descriptor;
    }

    int getOutputDim() const { return output_dim_; }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    int input_height_;
    int input_width_;
    int output_dim_;
};
#endif

int main(int argc, char* argv[]) {
    std::cout << "=== ONNX Runtime Global Feature Extraction ===\n" << std::endl;

#ifdef ONNXRUNTIME_AVAILABLE
    // Check arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model.onnx> [image1] [image2] ..."
                  << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " netvlad.onnx query.jpg db1.jpg db2.jpg"
                  << std::endl;
        std::cout << "\nTo create a model, run:" << std::endl;
        std::cout << "  python examples/export_onnx.py" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    try {
        // Load model
        std::cout << "Loading model: " << model_path << std::endl;
        ONNXFeatureExtractor extractor(model_path, 480, 640);

        std::cout << "Output dimension: " << extractor.getOutputDim() << std::endl;
        std::cout << std::endl;

        // Process images
        std::vector<std::vector<float>> descriptors;
        std::vector<std::string> image_paths;

        for (int i = 2; i < argc; ++i) {
            std::string image_path = argv[i];

            std::cout << "Processing: " << image_path << std::endl;

            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "  Failed to load image!" << std::endl;
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto descriptor = extractor.extractDescriptor(image);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count();

            std::cout << "  Extraction time: " << duration << " ms" << std::endl;

            descriptors.push_back(descriptor);
            image_paths.push_back(image_path);
        }

        // Compute pairwise similarities
        if (descriptors.size() >= 2) {
            std::cout << "\n=== Pairwise Similarities ===" << std::endl;

            for (size_t i = 0; i < descriptors.size(); ++i) {
                for (size_t j = i + 1; j < descriptors.size(); ++j) {
                    float sim = cosineSimilarity(descriptors[i], descriptors[j]);
                    std::cout << "  " << image_paths[i] << " <-> "
                              << image_paths[j] << ": " << sim << std::endl;
                }
            }
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }

#else
    std::cerr << "ONNX Runtime not available!" << std::endl;
    std::cerr << "Please build with -DONNXRUNTIME_ROOT=/path/to/onnxruntime"
              << std::endl;
    return 1;
#endif

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
