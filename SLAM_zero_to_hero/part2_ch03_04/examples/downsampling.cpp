// 출처: 임형태님 pcl_tutorial 코드
// https://github.com/LimHyungTae/pcl_tutorial

#include <pcl/filters/voxel_grid.h>

#include <sstream>

#include "kitti_cloud.hpp"

int main(int argc, char** argv) {
    std::cout << "=== VoxelGrid downsampling ===" << std::endl;

    const auto cloud = kitti::loadScan(argc, argv);
    if (!cloud) {
        return 1;
    }

    const float leaf_size_m = 0.5f;

    kitti::CloudT::Ptr ds_cloud(new kitti::CloudT);
    const double elapsed_ms = kitti::timeMs([&] {
        pcl::VoxelGrid<kitti::PointT> vox;
        vox.setInputCloud(cloud);
        vox.setLeafSize(leaf_size_m, leaf_size_m, leaf_size_m);
        vox.filter(*ds_cloud);
    });

    const double kept_percent =
        100.0 * static_cast<double>(ds_cloud->size()) / static_cast<double>(cloud->size());

    std::cout << "Leaf size:    " << leaf_size_m << " m" << std::endl;
    std::cout << "Input:        " << cloud->size() << " points" << std::endl;
    std::cout << "Downsampled:  " << ds_cloud->size() << " points (" << std::fixed
              << std::setprecision(1) << kept_percent << "% of the input)" << std::endl;
    std::cout << "Filter time:  " << std::fixed << std::setprecision(1) << elapsed_ms << " ms"
              << std::endl;

    if (!kitti::hasDisplay()) {
        kitti::reportNoDisplay();
        return 0;
    }

    const auto cloud_colored = kitti::colorized(*cloud, {0, 255, 0});
    const auto ds_colored = kitti::colorized(*ds_cloud, {0, 255, 0});

    pcl::visualization::PCLVisualizer viewer("VoxelGrid downsampling");
    int left = 0;
    int right = 0;
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, left);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, right);
    viewer.setBackgroundColor(0.05, 0.05, 0.05);

    std::ostringstream ds_label;
    ds_label << "VoxelGrid " << std::setprecision(2) << leaf_size_m << " m: "
             << ds_cloud->size() << " points";
    viewer.addText("Input: " + std::to_string(cloud->size()) + " points", 10, 10, 14, 1, 1, 1,
                   "label_input", left);
    viewer.addText(ds_label.str(), 10, 10, 14, 1, 1, 1, "label_ds", right);

    viewer.addPointCloud<pcl::PointXYZRGB>(cloud_colored, "input", left);
    viewer.addPointCloud<pcl::PointXYZRGB>(ds_colored, "downsampled", right);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
                                            "input");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
                                            "downsampled");
    kitti::applyView(viewer, kitti::View::topDown(45.0f));

    std::cout << "\nClose the viewer window to exit." << std::endl;
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }

    return 0;
}
