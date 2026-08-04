// 출처: 임형태님 pcl_tutorial 코드
// https://github.com/LimHyungTae/pcl_tutorial

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace kitti {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using ColorCloudT = pcl::PointCloud<pcl::PointXYZRGB>;

inline CloudT::Ptr loadBin(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return nullptr;
    }

    CloudT::Ptr cloud(new CloudT);
    float record[4];
    while (file.read(reinterpret_cast<char*>(record), sizeof(record))) {
        cloud->push_back(PointT(record[0], record[1], record[2]));
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

inline CloudT::Ptr loadScan(int argc, char** argv) {
    std::vector<std::string> candidates;
    if (argc > 1) {
        candidates.emplace_back(argv[1]);
    } else {
        candidates = {"data/000000.bin", "../data/000000.bin", "000000.bin"};
    }

    for (const auto& path : candidates) {
        if (!std::filesystem::exists(path)) {
            continue;
        }
        const auto cloud = loadBin(path);
        if (!cloud || cloud->empty()) {
            std::cerr << "Error: " << path << " is not a readable KITTI .bin scan" << std::endl;
            return nullptr;
        }
        std::cout << "Loaded " << path << ": " << cloud->size() << " points" << std::endl;
        return cloud;
    }

    std::cerr << "Error: could not find a KITTI scan." << std::endl;
    std::cerr << "Pass one explicitly (" << argv[0] << " <scan.bin>), or run from the"
              << " exercise root so data/000000.bin resolves." << std::endl;
    return nullptr;
}

inline void colorize(const CloudT& cloud, ColorCloudT& colored,
                     const std::vector<int>& color) {
    colored.clear();
    colored.reserve(cloud.size());

    pcl::PointXYZRGB pt_rgb;
    for (const auto& pt : cloud.points) {
        pt_rgb.x = pt.x;
        pt_rgb.y = pt.y;
        pt_rgb.z = pt.z;
        pt_rgb.r = static_cast<std::uint8_t>(color[0]);
        pt_rgb.g = static_cast<std::uint8_t>(color[1]);
        pt_rgb.b = static_cast<std::uint8_t>(color[2]);
        colored.points.emplace_back(pt_rgb);
    }

    colored.width = colored.size();
    colored.height = 1;
}

inline ColorCloudT::Ptr colorized(const CloudT& cloud, const std::vector<int>& color) {
    ColorCloudT::Ptr colored(new ColorCloudT);
    colorize(cloud, *colored, color);
    return colored;
}

template <typename Func>
inline double timeMs(Func&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline bool hasDisplay() { return std::getenv("DISPLAY") != nullptr; }

inline void reportNoDisplay() {
    std::cout << "\nDISPLAY is not set — skipping the viewer window." << std::endl;
    std::cout << "Re-run with a display to see the point clouds:" << std::endl;
    std::cout << "  docker run -it --rm -e DISPLAY=$DISPLAY"
              << " -v /tmp/.X11-unix:/tmp/.X11-unix ..." << std::endl;
}

struct View {
    float pos[3];
    float focal[3];
    float up[3];

    static View chase() { return {{-28.0f, 0.0f, 20.0f}, {12.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}; }

    static View topDown(float height_m, float center_x_m = 0.0f, float center_y_m = 0.0f) {
        return {{center_x_m, center_y_m, height_m},
                {center_x_m, center_y_m, 0.0f},
                {1.0f, 0.0f, 0.0f}};
    }
};

inline void applyView(pcl::visualization::PCLVisualizer& viewer, const View& view) {
    viewer.setCameraPosition(view.pos[0], view.pos[1], view.pos[2], view.focal[0],
                             view.focal[1], view.focal[2], view.up[0], view.up[1], view.up[2]);
}

struct Layer {
    ColorCloudT::Ptr cloud;
    std::string id;
    int point_size = 2;
};

inline void show(const std::string& title, const std::vector<Layer>& layers,
                 const View& view = View::chase()) {
    if (!hasDisplay()) {
        reportNoDisplay();
        return;
    }

    pcl::visualization::PCLVisualizer viewer(title);
    viewer.setBackgroundColor(0.05, 0.05, 0.05);
    for (const auto& layer : layers) {
        viewer.addPointCloud<pcl::PointXYZRGB>(layer.cloud, layer.id);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, layer.point_size, layer.id);
    }
    viewer.addCoordinateSystem(3.0);
    applyView(viewer, view);

    std::cout << "\nClose the viewer window to exit." << std::endl;
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

}  // namespace kitti
