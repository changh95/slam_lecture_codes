/*
 * TensorRT Logger Implementation
 *
 * Defines the global logger instance used throughout the application.
 */

#include "logging.h"

namespace tensorrt_log {

// Global logger instance with default WARNING severity
Logger gLogger(Logger::Severity::kWARNING);

}  // namespace tensorrt_log
