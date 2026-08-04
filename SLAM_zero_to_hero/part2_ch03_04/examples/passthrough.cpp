// 출처: 임형태님 pcl_tutorial 코드
// https://github.com/LimHyungTae/pcl_tutorial

#include <pcl/filters/passthrough.h>

#include "kitti_cloud.hpp"

int main(int argc, char** argv) {
    std::cout << "=== PassThrough filter: remove the ego-vehicle ===" << std::endl;

    const auto cloud = kitti::loadScan(argc, argv);
    if (!cloud) {
        return 1;
    }

    const float car_size_m = 3.0f;

    kitti::CloudT::Ptr center(new kitti::CloudT);
    kitti::CloudT::Ptr outskirt(new kitti::CloudT);
    kitti::CloudT::Ptr output(new kitti::CloudT);

    const double elapsed_ms = kitti::timeMs([&] {
        pcl::PassThrough<kitti::PointT> xfilter;
        xfilter.setInputCloud(cloud);
        xfilter.setFilterFieldName("x");
        xfilter.setFilterLimits(-car_size_m, car_size_m);
        xfilter.filter(*center);
        xfilter.setNegative(true);
        xfilter.filter(*outskirt);

        pcl::PassThrough<kitti::PointT> yfilter;
        yfilter.setInputCloud(center);
        yfilter.setFilterFieldName("y");
        yfilter.setFilterLimits(-car_size_m, car_size_m);
        yfilter.setNegative(true);
        yfilter.filter(*output);

        *output += *outskirt;
    });

    std::cout << "Box removed:  |x| < " << car_size_m << " m and |y| < " << car_size_m << " m"
              << std::endl;
    std::cout << "Input:        " << cloud->size() << " points" << std::endl;
    std::cout << "Kept:         " << output->size() << " points" << std::endl;
    std::cout << "Removed:      " << cloud->size() - output->size() << " points" << std::endl;
    std::cout << "Filter time:  " << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
              << std::endl;

    kitti::show("PassThrough filter (red: input, green: kept)",
                {{kitti::colorized(*cloud, {255, 0, 0}), "input", 3},
                 {kitti::colorized(*output, {0, 255, 0}), "kept", 3}},
                kitti::View::topDown(35.0f));

    return 0;
}
