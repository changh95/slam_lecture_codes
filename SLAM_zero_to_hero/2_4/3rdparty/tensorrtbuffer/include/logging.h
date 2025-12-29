/*
 * TensorRT Logging Utilities
 *
 * Provides a logger implementation for TensorRT that controls
 * verbosity and message formatting.
 *
 * Based on NVIDIA TensorRT samples.
 */

#ifndef TENSORRT_LOGGING_H
#define TENSORRT_LOGGING_H

#include <iostream>
#include <string>
#include <NvInfer.h>

namespace tensorrt_log {

/**
 * @brief TensorRT logger implementation
 *
 * TensorRT uses this logger for all internal messages. Control
 * verbosity by setting the reportable severity level.
 */
class Logger : public nvinfer1::ILogger {
public:
    enum class Severity : int {
        kINTERNAL_ERROR = 0,  // Internal error (should never happen)
        kERROR = 1,           // Error in user code or TensorRT
        kWARNING = 2,         // Warning (may indicate problem)
        kINFO = 3,            // Informational message
        kVERBOSE = 4          // Verbose debug info
    };

    Logger(Severity severity = Severity::kWARNING)
        : mReportableSeverity(severity) {}

    /**
     * @brief Log a message from TensorRT
     *
     * Only logs messages with severity <= reportable severity.
     */
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // Suppress messages below threshold
        if (static_cast<int>(severity) > static_cast<int>(mReportableSeverity)) {
            return;
        }

        // Format message with severity prefix
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "[TRT INTERNAL ERROR] " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "[TRT ERROR] " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "[TRT WARNING] " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cout << "[TRT INFO] " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kVERBOSE:
                std::cout << "[TRT VERBOSE] " << msg << std::endl;
                break;
        }
    }

    nvinfer1::ILogger& getTRTLogger() {
        return *this;
    }

    void setReportableSeverity(Severity severity) {
        mReportableSeverity = severity;
    }

    Severity getReportableSeverity() const {
        return mReportableSeverity;
    }

private:
    Severity mReportableSeverity;
};

// Global logger instance
extern Logger gLogger;

// Helper function to set severity
inline void setReportableSeverity(Logger::Severity severity) {
    gLogger.setReportableSeverity(severity);
}

}  // namespace tensorrt_log

#endif  // TENSORRT_LOGGING_H
