// 출처: 임형태님 pcl_tutorial 코드
// https://github.com/LimHyungTae/pcl_tutorial

#include <pcl/kdtree/kdtree_flann.h>

#include <cmath>

#include "kitti_cloud.hpp"

int main(int argc, char** argv) {
    std::cout << "=== KD-tree search (KNN + radius) ===" << std::endl;

    const auto cloud = kitti::loadScan(argc, argv);
    if (!cloud) {
        return 1;
    }

    kitti::PointT search_point;
    search_point.x = 8.0f;
    search_point.y = 10.0f;
    search_point.z = 0.1f;

    const int k = 100;
    const float radius_m = 5.0f;

    pcl::KdTreeFLANN<kitti::PointT> kdtree;
    const double build_ms = kitti::timeMs([&] { kdtree.setInputCloud(cloud); });

    std::vector<int> knn_indices;
    std::vector<float> knn_sqr_dists;
    const double knn_ms = kitti::timeMs(
        [&] { kdtree.nearestKSearch(search_point, k, knn_indices, knn_sqr_dists); });

    kitti::CloudT::Ptr knn_result(new kitti::CloudT);
    for (const auto& idx : knn_indices) {
        knn_result->push_back((*cloud)[idx]);
    }

    std::vector<int> radius_indices;
    std::vector<float> radius_sqr_dists;
    const double radius_ms = kitti::timeMs(
        [&] { kdtree.radiusSearch(search_point, radius_m, radius_indices, radius_sqr_dists); });

    kitti::CloudT::Ptr radius_result(new kitti::CloudT);
    for (const auto& idx : radius_indices) {
        radius_result->push_back((*cloud)[idx]);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Search point: (" << search_point.x << ", " << search_point.y << ", "
              << search_point.z << ")" << std::endl;
    std::cout << "Tree build:   " << build_ms << " ms for " << cloud->size() << " points"
              << std::endl;

    std::cout << "\nnearestKSearch, K=" << k << ": " << knn_result->size() << " points in "
              << knn_ms << " ms" << std::endl;
    if (!knn_sqr_dists.empty()) {
        std::cout << "  closest neighbour:  " << std::sqrt(knn_sqr_dists.front()) << " m"
                  << std::endl;
        std::cout << "  farthest of the K:  " << std::sqrt(knn_sqr_dists.back()) << " m"
                  << std::endl;
    }

    std::cout << "\nradiusSearch, r=" << radius_m << " m: " << radius_result->size()
              << " points in " << radius_ms << " ms" << std::endl;
    if (!radius_sqr_dists.empty()) {
        std::cout << "  closest neighbour:  " << std::sqrt(radius_sqr_dists.front()) << " m"
                  << std::endl;
        std::cout << "  farthest returned:  " << std::sqrt(radius_sqr_dists.back()) << " m"
                  << std::endl;
    }

    kitti::show("KD-tree search (white: scan, red: radius, green: KNN)",
                {{kitti::colorized(*cloud, {180, 180, 180}), "scan", 2},
                 {kitti::colorized(*radius_result, {255, 0, 0}), "radius_search", 3},
                 {kitti::colorized(*knn_result, {0, 255, 0}), "knn_search", 7}},
                kitti::View::topDown(30.0f, search_point.x, search_point.y));

    return 0;
}
