/**
 * Shared helpers for the part2_ch03_06 ICP demos:
 *  - point cloud loading from .ply, .pcd, or KITTI velodyne .bin files
 *  - locating the bundled data/ files regardless of the working directory
 *  - scale-relative ICP parameters derived from the cloud's bounding box
 *
 * The ICP demos run on the Stanford bunny (data/bun_zipper_res3.ply) by
 * default. The bunny is only ~0.25 m across, so absolute parameters tuned for
 * LiDAR scans (0.5 m correspondence distance, 0.1 m normal radius) are larger
 * than the whole model and make ICP meaningless. Deriving them from the
 * bounding-box diagonal keeps one set of demos working for both the bunny and
 * room- or street-scale clouds.
 */

#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/console/print.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

namespace demo {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/** Relative filename of the bunny model shipped with this chapter */
inline const char* kBunnyFile = "bun_zipper_res3.ply";

/**
 * @brief Load a KITTI velodyne scan (.bin): raw float32 records of x,y,z,intensity
 */
inline CloudT::Ptr loadKittiBin(const std::string& file)
{
    std::ifstream input(file, std::ios::binary);
    if (!input)
    {
        return nullptr;
    }

    CloudT::Ptr cloud(new CloudT);
    float record[4];
    while (input.read(reinterpret_cast<char*>(record), sizeof(record)))
    {
        cloud->push_back(PointT(record[0], record[1], record[2]));
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * @brief Load a point cloud from a .ply, .pcd, or KITTI .bin file
 * @return nullptr on failure
 */
inline CloudT::Ptr loadCloud(const std::string& file)
{
    const std::string ext = std::filesystem::path(file).extension().string();

    if (ext == ".bin")
    {
        CloudT::Ptr cloud = loadKittiBin(file);
        return (cloud && !cloud->empty()) ? cloud : nullptr;
    }

    CloudT::Ptr cloud(new CloudT);
    int ret = -1;

    if (ext == ".ply")
    {
        // The bunny .ply carries mesh faces as well as vertices, and the PLY
        // reader warns about every property it cannot map onto PointXYZ. Only
        // the vertices are wanted here, so keep that warning out of the output.
        const auto previous = pcl::console::getVerbosityLevel();
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
        ret = pcl::io::loadPLYFile<PointT>(file, *cloud);
        pcl::console::setVerbosityLevel(previous);
    }
    else
    {
        ret = pcl::io::loadPCDFile<PointT>(file, *cloud);
    }

    if (ret == -1 || cloud->empty())
    {
        return nullptr;
    }
    return cloud;
}

/**
 * @brief Resolve a file in this chapter's data/ directory
 *
 * The demos are run both from the project root (./build/icp_basic) and from
 * inside build/ (the Docker WORKDIR), and data/ is mounted at /data in the
 * container, so try each location.
 * @return empty string if the file was not found
 */
inline std::string findDataFile(const std::string& filename)
{
    const std::vector<std::string> candidates = {
        "data/" + filename,
        "../data/" + filename,
        "../../data/" + filename,
        "/data/" + filename,
    };

    for (const auto& path : candidates)
    {
        if (std::filesystem::exists(path))
        {
            return path;
        }
    }
    return {};
}

/**
 * @brief Diagonal length of the cloud's axis-aligned bounding box
 *
 * Used as the characteristic scale of the model: ICP distances, voxel sizes and
 * viewer geometry are expressed as fractions of it.
 */
inline double bboxDiagonal(const CloudT& cloud)
{
    PointT min_pt, max_pt;
    pcl::getMinMax3D(cloud, min_pt, max_pt);
    return (max_pt.getVector3fMap() - min_pt.getVector3fMap()).norm();
}

/**
 * @brief Move the cloud's centroid to the origin
 *
 * The bunny sits offset from the origin, so rotating it about the origin
 * displaces it far more than the same rotation about its own centre. Centring
 * first makes the injected rotation a pure rotation of the model, and lets the
 * viewer point its camera at the origin.
 */
inline CloudT::Ptr centerCloud(const CloudT& cloud)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(cloud, centroid);

    Eigen::Affine3f shift = Eigen::Affine3f::Identity();
    shift.translation() = -centroid.head<3>();

    CloudT::Ptr out(new CloudT);
    pcl::transformPointCloud(cloud, *out, shift);
    return out;
}

/**
 * @brief Build a rigid transform from a translation and XYZ rotation (radians)
 */
inline Eigen::Matrix4f makeTransform(float tx, float ty, float tz,
                                     float rx, float ry, float rz)
{
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
    transform.translation() << tx, ty, tz;
    return transform.matrix();
}

/**
 * @brief Voxel-downsample a cloud (leaf size in meters)
 */
inline CloudT::Ptr voxelDownsample(const CloudT& cloud, float leaf)
{
    CloudT::Ptr out(new CloudT);
    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(cloud.makeShared());
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*out);
    return out;
}

/**
 * @brief Print a 4x4 transform as a rotation matrix plus translation
 */
inline void printTransformation(const Eigen::Matrix4f& T, const std::string& name)
{
    std::cout << "\n" << name << ":\n";
    std::cout << "  Rotation matrix:\n";
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "    [";
        for (int j = 0; j < 3; ++j)
        {
            std::cout << std::fixed << std::setprecision(6) << std::setw(10) << T(i, j);
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "  Translation: [" << std::fixed << std::setprecision(6)
              << T(0, 3) << ", " << T(1, 3) << ", " << T(2, 3) << "]\n";
}

/**
 * @brief Rotation and translation error of an estimate against ground truth
 *
 * The demos build the source cloud by applying a known transform to the target,
 * so the exact answer is available and ICP can be scored against it instead of
 * only reporting its own fitness score.
 */
struct PoseError
{
    double rotation_deg;
    double translation_m;
};

inline PoseError poseError(const Eigen::Matrix4f& estimated,
                           const Eigen::Matrix4f& ground_truth)
{
    const Eigen::Matrix4f delta = ground_truth.inverse() * estimated;

    // Angle of the residual rotation.
    //
    // Do NOT use the textbook acos((trace - 1) / 2) here. Neither input is
    // exactly orthonormal - the KITTI ground truth is rebased through Tr and
    // round-tripped via text, and the estimate is a long product of float
    // matrices - so the residual trace lands slightly above 3, the argument
    // clamps to 1, and acos returns exactly 0. That silently reports "no
    // rotation error" precisely in the sub-degree range that matters. Going
    // through a normalized quaternion absorbs the non-orthonormality, and the
    // atan2 form stays accurate for both tiny and large angles.
    Eigen::Quaterniond q(delta.block<3, 3>(0, 0).cast<double>());
    q.normalize();
    const double angle = 2.0 * std::atan2(q.vec().norm(), std::abs(q.w()));

    return {angle * 180.0 / M_PI,
            delta.block<3, 1>(0, 3).norm()};
}

/**
 * @brief Report how well ICP recovered a known transform
 */
inline void printPoseError(const Eigen::Matrix4f& estimated,
                           const Eigen::Matrix4f& ground_truth,
                           double model_scale)
{
    const PoseError err = poseError(estimated, ground_truth);

    std::cout << "\n--- Accuracy vs Ground Truth ---\n";
    std::cout << "  Rotation error:    " << std::fixed << std::setprecision(4)
              << err.rotation_deg << " deg\n";
    std::cout << "  Translation error: " << std::setprecision(6)
              << err.translation_m << " m ("
              << std::setprecision(3) << 100.0 * err.translation_m / model_scale
              << " % of model size)\n";
}

}  // namespace demo
