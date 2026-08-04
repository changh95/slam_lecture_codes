#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <cmath>

#include "kitti_cloud.hpp"

int main(int argc, char** argv) {
    std::cout << "=== RANSAC plane detection (ground plane) ===" << std::endl;

    const auto cloud = kitti::loadScan(argc, argv);
    if (!cloud) {
        return 1;
    }

    const double distance_threshold_m = 0.2;

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    const double elapsed_ms = kitti::timeMs([&] {
        pcl::SACSegmentation<kitti::PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_threshold_m);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
    });

    if (inliers->indices.empty()) {
        PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
        return 1;
    }

    kitti::CloudT::Ptr plane(new kitti::CloudT);
    kitti::CloudT::Ptr rest(new kitti::CloudT);
    std::vector<bool> is_inlier(cloud->size(), false);
    for (const auto& idx : inliers->indices) {
        is_inlier[idx] = true;
        plane->push_back((*cloud)[idx]);
    }
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        if (!is_inlier[i]) {
            rest->push_back((*cloud)[i]);
        }
    }

    const float a = coefficients->values[0];
    const float b = coefficients->values[1];
    const float c = coefficients->values[2];
    const float d = coefficients->values[3];
    const double tilt_from_z_deg = std::acos(std::min(1.0f, std::abs(c))) * 180.0 / M_PI;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Threshold:    " << distance_threshold_m << " m" << std::endl;
    std::cout << "Model:        " << a << "x + " << b << "y + " << c << "z + " << d << " = 0"
              << std::endl;
    std::cout << "Tilt from z:  " << tilt_from_z_deg << " deg" << std::endl;
    std::cout << "Sensor height above the plane: " << std::abs(d) << " m" << std::endl;
    std::cout << "Input:        " << cloud->size() << " points" << std::endl;
    std::cout << "Plane:        " << plane->size() << " points" << std::endl;
    std::cout << "Off-plane:    " << rest->size() << " points" << std::endl;
    std::cout << "Segment time: " << std::setprecision(1) << elapsed_ms << " ms" << std::endl;

    kitti::show("RANSAC plane detection (red: plane, green: rest)",
                {{kitti::colorized(*rest, {0, 255, 0}), "rest", 2},
                 {kitti::colorized(*plane, {255, 0, 0}), "plane", 2}},
                kitti::View::chase());

    return 0;
}
