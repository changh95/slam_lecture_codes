#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>

#include <cmath>

#include "kitti_cloud.hpp"

int main(int argc, char** argv) {
    std::cout << "=== Normal estimation ===" << std::endl;

    const auto cloud = kitti::loadScan(argc, argv);
    if (!cloud) {
        return 1;
    }

    const float leaf_size_m = 0.4f;
    const double search_radius_m = 1.0;

    kitti::CloudT::Ptr ds_cloud(new kitti::CloudT);
    pcl::VoxelGrid<kitti::PointT> vox;
    vox.setInputCloud(cloud);
    vox.setLeafSize(leaf_size_m, leaf_size_m, leaf_size_m);
    vox.filter(*ds_cloud);

    std::cout << "Downsampled to " << ds_cloud->size() << " points (" << leaf_size_m
              << " m voxels)" << std::endl;
    std::cout << "Search radius: " << search_radius_m << " m" << std::endl;
    std::cout << "Estimating normals..." << std::endl;

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    const double serial_ms = kitti::timeMs([&] {
        pcl::NormalEstimation<kitti::PointT, pcl::Normal> ne;
        ne.setInputCloud(ds_cloud);
        ne.setSearchMethod(
            pcl::search::KdTree<kitti::PointT>::Ptr(new pcl::search::KdTree<kitti::PointT>));
        ne.setRadiusSearch(search_radius_m);
        ne.setViewPoint(0.0f, 0.0f, 0.0f);
        ne.compute(*normals);
    });

    pcl::PointCloud<pcl::Normal>::Ptr normals_omp(new pcl::PointCloud<pcl::Normal>);
    const double omp_ms = kitti::timeMs([&] {
        pcl::NormalEstimationOMP<kitti::PointT, pcl::Normal> ne;
        ne.setInputCloud(ds_cloud);
        ne.setSearchMethod(
            pcl::search::KdTree<kitti::PointT>::Ptr(new pcl::search::KdTree<kitti::PointT>));
        ne.setRadiusSearch(search_radius_m);
        ne.setViewPoint(0.0f, 0.0f, 0.0f);
        ne.compute(*normals_omp);
    });

    std::size_t valid = 0;
    std::size_t horizontal = 0;
    for (const auto& n : normals->points) {
        if (!std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) ||
            !std::isfinite(n.normal_z)) {
            continue;
        }
        ++valid;
        if (std::abs(n.normal_z) > 0.9f) {
            ++horizontal;
        }
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\nNormalEstimation:     " << serial_ms << " ms" << std::endl;
    std::cout << "NormalEstimationOMP:  " << omp_ms << " ms";
    if (omp_ms > 0.0) {
        std::cout << "  (" << std::setprecision(2) << serial_ms / omp_ms << "x speedup)";
    }
    std::cout << std::endl;

    std::cout << std::setprecision(1);
    std::cout << "\nValid normals:            " << valid << " / " << normals->size()
              << std::endl;
    std::cout << "NaN (too few neighbours): " << normals->size() - valid << std::endl;
    std::cout << "Horizontal surfaces (|n_z| > 0.9): " << horizontal << " points ("
              << 100.0 * static_cast<double>(horizontal) /
                     static_cast<double>(std::max<std::size_t>(valid, 1))
              << "% of valid)" << std::endl;

    if (!kitti::hasDisplay()) {
        kitti::reportNoDisplay();
        return 0;
    }

    const int normal_draw_stride = 10;
    const float normal_draw_length_m = 0.5f;
    const auto ds_colored = kitti::colorized(*ds_cloud, {0, 255, 0});

    pcl::visualization::PCLVisualizer viewer("Normal estimation (every 10th normal drawn)");
    viewer.setBackgroundColor(0.05, 0.05, 0.05);
    viewer.addPointCloud<pcl::PointXYZRGB>(ds_colored, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
                                            "cloud");
    viewer.addPointCloudNormals<kitti::PointT, pcl::Normal>(
        ds_cloud, normals, normal_draw_stride, normal_draw_length_m, "normals");
    viewer.addCoordinateSystem(3.0);
    kitti::applyView(viewer, {{-24.0f, 0.0f, 15.0f}, {10.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});

    std::cout << "\nClose the viewer window to exit." << std::endl;
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }

    return 0;
}
