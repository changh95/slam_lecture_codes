#ifndef MONOCULAR_VO_HPP
#define MONOCULAR_VO_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

/**
 * @brief Monocular Visual Odometry class
 *
 * Implements a complete monocular VO pipeline:
 * 1. Feature detection (FAST/GFTT)
 * 2. Feature tracking (KLT optical flow)
 * 3. Essential matrix estimation
 * 4. Pose recovery
 * 5. Trajectory accumulation
 */
class MonocularVO {
public:
    /**
     * @brief Constructor
     * @param focal Focal length in pixels
     * @param pp Principal point (cx, cy)
     * @param min_features Minimum features to maintain
     * @param max_features Maximum features to detect
     */
    MonocularVO(double focal, cv::Point2d pp,
                int min_features = 100, int max_features = 2000);

    virtual ~MonocularVO() = default;

    /**
     * @brief Process a new frame
     * @param frame Input BGR image
     * @return true if pose was successfully estimated
     */
    bool processFrame(const cv::Mat& frame);

    /**
     * @brief Get current position in world frame
     * @return 3x1 translation vector
     */
    cv::Mat getPosition() const { return t_total_.clone(); }

    /**
     * @brief Get current rotation in world frame
     * @return 3x3 rotation matrix
     */
    cv::Mat getRotation() const { return R_total_.clone(); }

    /**
     * @brief Get current position as Point3d
     */
    cv::Point3d getPositionPoint() const;

    /**
     * @brief Get frame count
     */
    int getFrameId() const { return frame_id_; }

    /**
     * @brief Get number of tracked features
     */
    size_t getFeatureCount() const { return prev_pts_.size(); }

    /**
     * @brief Get current tracked features for visualization
     */
    std::vector<cv::Point2f> getCurrentFeatures() const { return prev_pts_; }

    /**
     * @brief Get trajectory history
     */
    const std::vector<cv::Point3d>& getTrajectory() const { return trajectory_; }

    /**
     * @brief Enable/disable visualization output
     */
    void setVisualization(bool enable) { visualize_ = enable; }

    /**
     * @brief Get visualization frame
     */
    cv::Mat getVisualizationFrame() const { return vis_frame_.clone(); }

protected:
    /**
     * @brief Detect features in the image
     * @param gray Grayscale image
     * @param pts Output feature points
     */
    void detectFeatures(const cv::Mat& gray, std::vector<cv::Point2f>& pts);

    /**
     * @brief Track features from previous to current frame
     * @param prev Previous grayscale image
     * @param curr Current grayscale image
     * @param prev_pts Previous feature points
     * @param curr_pts Output tracked points
     * @param status Output tracking status
     */
    void trackFeatures(const cv::Mat& prev, const cv::Mat& curr,
                       const std::vector<cv::Point2f>& prev_pts,
                       std::vector<cv::Point2f>& curr_pts,
                       std::vector<uchar>& status);

    /**
     * @brief Perform bidirectional tracking for outlier rejection
     */
    void bidirectionalTracking(const cv::Mat& prev, const cv::Mat& curr,
                               std::vector<cv::Point2f>& prev_pts,
                               std::vector<cv::Point2f>& curr_pts,
                               std::vector<uchar>& status);

    /**
     * @brief Get scale factor (override for ground truth or sensor fusion)
     * @return Scale factor for translation
     */
    virtual double getScale();

    /**
     * @brief Create visualization frame
     */
    void createVisualization(const cv::Mat& frame,
                             const std::vector<cv::Point2f>& prev_pts,
                             const std::vector<cv::Point2f>& curr_pts,
                             const std::vector<uchar>& status);

    // Camera parameters
    double focal_;
    cv::Point2d pp_;
    cv::Mat K_;  // Camera matrix

    // Feature detection parameters
    int min_features_;
    int max_features_;

    // State
    int frame_id_;
    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_pts_;

    // Accumulated pose
    cv::Mat R_total_;  // World rotation
    cv::Mat t_total_;  // World position

    // Trajectory history
    std::vector<cv::Point3d> trajectory_;

    // Visualization
    bool visualize_;
    cv::Mat vis_frame_;
};

/**
 * @brief Monocular VO with KITTI ground truth scale
 *
 * Uses ground truth poses to obtain metric scale for evaluation.
 */
class MonocularVO_KITTI : public MonocularVO {
public:
    /**
     * @brief Constructor
     * @param focal Focal length in pixels
     * @param pp Principal point
     * @param poses_file Path to KITTI poses file (e.g., 00.txt)
     */
    MonocularVO_KITTI(double focal, cv::Point2d pp,
                      const std::string& poses_file);

    /**
     * @brief Get ground truth trajectory
     */
    const std::vector<cv::Point3d>& getGroundTruth() const { return ground_truth_; }

    /**
     * @brief Compute Absolute Trajectory Error (RMSE)
     */
    double computeATE() const;

protected:
    /**
     * @brief Load ground truth poses from KITTI format file
     */
    void loadGroundTruth(const std::string& file);

    /**
     * @brief Get scale from ground truth
     */
    double getScale() override;

    std::vector<cv::Point3d> ground_truth_;
};

/**
 * @brief Utility functions for visualization
 */
namespace vo_utils {

/**
 * @brief Draw trajectory on a canvas
 * @param estimated Estimated trajectory
 * @param ground_truth Ground truth trajectory (optional)
 * @param scale Visualization scale
 * @return Visualization image
 */
cv::Mat drawTrajectory(const std::vector<cv::Point3d>& estimated,
                       const std::vector<cv::Point3d>& ground_truth = {},
                       double scale = 1.0);

/**
 * @brief Draw feature tracks on frame
 */
cv::Mat drawFeatureTracks(const cv::Mat& frame,
                          const std::vector<cv::Point2f>& prev_pts,
                          const std::vector<cv::Point2f>& curr_pts,
                          const std::vector<uchar>& status);

/**
 * @brief Load image sequence from directory
 * @param dir Directory path
 * @param start_idx Start index
 * @param end_idx End index (-1 for all)
 * @return Vector of image paths
 */
std::vector<std::string> loadImageSequence(const std::string& dir,
                                            int start_idx = 0,
                                            int end_idx = -1);

}  // namespace vo_utils

#endif  // MONOCULAR_VO_HPP
