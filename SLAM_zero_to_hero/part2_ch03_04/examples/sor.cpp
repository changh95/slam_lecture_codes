// 출처: 임형태님 pcl_tutorial 코드
// https://github.com/LimHyungTae/pcl_tutorial

#include <pcl/filters/statistical_outlier_removal.h>

#include <cmath>

#include "kitti_cloud.hpp"

namespace {

double meanRangeM(const kitti::CloudT& cloud) {
    if (cloud.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& pt : cloud.points) {
        sum += std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    }
    return sum / static_cast<double>(cloud.size());
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "=== Statistical Outlier Removal ===" << std::endl;

    const auto cloud = kitti::loadScan(argc, argv);
    if (!cloud) {
        return 1;
    }

    const int mean_k = 200;
    const double stddev_mul = 5.0;

    kitti::CloudT::Ptr inliers(new kitti::CloudT);
    kitti::CloudT::Ptr outliers(new kitti::CloudT);

    std::cout << "Running SOR on " << cloud->size() << " points (this takes a few seconds)..."
              << std::endl;

    const double elapsed_ms = kitti::timeMs([&] {
        pcl::StatisticalOutlierRemoval<kitti::PointT> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(mean_k);
        sor.setStddevMulThresh(stddev_mul);
        sor.filter(*inliers);

        sor.setNegative(true);
        sor.filter(*outliers);
    });

    std::cout << "MeanK:        " << mean_k << " neighbours" << std::endl;
    std::cout << "Threshold:    " << stddev_mul << " sigma" << std::endl;
    std::cout << "Input:        " << cloud->size() << " points" << std::endl;
    std::cout << "Inliers:      " << inliers->size() << " points" << std::endl;
    std::cout << "Outliers:     " << outliers->size() << " points" << std::endl;
    std::cout << "Filter time:  " << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
              << std::endl;
    std::cout << "Mean range, inliers:  " << meanRangeM(*inliers) << " m" << std::endl;
    std::cout << "Mean range, outliers: " << meanRangeM(*outliers) << " m" << std::endl;

    kitti::show("Statistical Outlier Removal (green: inliers, red: outliers)",
                {{kitti::colorized(*inliers, {40, 110, 40}), "inliers", 1},
                 {kitti::colorized(*outliers, {255, 0, 0}), "outliers", 5}},
                kitti::View::topDown(115.0f));

    return 0;
}
