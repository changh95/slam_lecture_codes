/**
 * Point Cloud Filtering with PCL
 *
 * This example demonstrates:
 * 1. VoxelGrid downsampling - Reduce point density
 * 2. StatisticalOutlierRemoval - Remove noise based on statistics
 * 3. RadiusOutlierRemoval - Remove points with few neighbors
 * 4. PassThrough filter - Crop by coordinate range
 *
 * These filters are essential for preprocessing point clouds in SLAM pipelines.
 */

#include <iostream>
#include <string>
#include <random>
#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>

using namespace std;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate a noisy synthetic point cloud
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateNoisyCloud(int num_points,
                                                         float noise_ratio = 0.1f) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    mt19937 rng(42);
    uniform_real_distribution<float> pos_dist(-2.0f, 2.0f);
    uniform_real_distribution<float> noise_dist(-10.0f, 10.0f);

    int main_points = static_cast<int>(num_points * (1.0f - noise_ratio));
    int noise_points = num_points - main_points;

    // Add main points (structured data)
    for (int i = 0; i < main_points; ++i) {
        pcl::PointXYZ point;
        point.x = pos_dist(rng);
        point.y = pos_dist(rng);
        point.z = pos_dist(rng);
        cloud->points.push_back(point);
    }

    // Add noise points (outliers)
    for (int i = 0; i < noise_points; ++i) {
        pcl::PointXYZ point;
        point.x = noise_dist(rng);
        point.y = noise_dist(rng);
        point.z = noise_dist(rng);
        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Time a function and return elapsed milliseconds
 */
template<typename Func>
double timeFunction(Func&& func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

// =============================================================================
// Filter Demonstrations
// =============================================================================

void demoVoxelGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                   const string& output_dir) {
    cout << "=== VoxelGrid Downsampling ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    // Different voxel sizes to demonstrate effect
    vector<float> voxel_sizes = {0.1f, 0.2f, 0.5f};

    for (float voxel_size : voxel_sizes) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(voxel_size, voxel_size, voxel_size);

        double time_ms = timeFunction([&]() {
            voxel.filter(*cloud_filtered);
        });

        float reduction = 100.0f * (1.0f - static_cast<float>(cloud_filtered->points.size()) /
                                           static_cast<float>(cloud->points.size()));

        cout << "  Voxel size " << voxel_size << "m: "
             << cloud->points.size() << " -> "
             << cloud_filtered->points.size() << " points ("
             << reduction << "% reduction, " << time_ms << " ms)" << endl;

        // Save the filtered cloud
        string filename = output_dir + "/voxel_" +
                          to_string(static_cast<int>(voxel_size * 100)) + "cm.pcd";
        pcl::io::savePCDFileBinary(filename, *cloud_filtered);
    }

    cout << endl;
}

void demoStatisticalOutlierRemoval(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                    const string& output_dir) {
    cout << "=== Statistical Outlier Removal ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outliers(
        new pcl::PointCloud<pcl::PointXYZ>);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);              // Number of neighbors to analyze
    sor.setStddevMulThresh(1.0);   // Std dev multiplier threshold

    double time_ms = timeFunction([&]() {
        sor.filter(*cloud_filtered);
    });

    // Get outliers
    sor.setNegative(true);
    sor.filter(*cloud_outliers);

    cout << "  Parameters: MeanK=50, StdDevThresh=1.0" << endl;
    cout << "  Inliers: " << cloud_filtered->points.size() << " points" << endl;
    cout << "  Outliers: " << cloud_outliers->points.size() << " points" << endl;
    cout << "  Time: " << time_ms << " ms" << endl;

    // Save results
    pcl::io::savePCDFileBinary(output_dir + "/sor_inliers.pcd", *cloud_filtered);
    pcl::io::savePCDFileBinary(output_dir + "/sor_outliers.pcd", *cloud_outliers);

    cout << endl;
}

void demoRadiusOutlierRemoval(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                               const string& output_dir) {
    cout << "=== Radius Outlier Removal ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZ>);

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(cloud);
    ror.setRadiusSearch(0.5);        // Search radius
    ror.setMinNeighborsInRadius(5);  // Minimum neighbors required

    double time_ms = timeFunction([&]() {
        ror.filter(*cloud_filtered);
    });

    cout << "  Parameters: Radius=0.5m, MinNeighbors=5" << endl;
    cout << "  Remaining: " << cloud_filtered->points.size() << " points" << endl;
    cout << "  Removed: " << (cloud->points.size() - cloud_filtered->points.size()) << " points" << endl;
    cout << "  Time: " << time_ms << " ms" << endl;

    pcl::io::savePCDFileBinary(output_dir + "/ror_filtered.pcd", *cloud_filtered);

    cout << endl;
}

void demoPassThrough(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                      const string& output_dir) {
    cout << "=== PassThrough Filter ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    // Filter on Z axis
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.0, 1.0);  // Keep z in [-1, 1]

        double time_ms = timeFunction([&]() {
            pass.filter(*cloud_filtered);
        });

        cout << "  Z filter [-1.0, 1.0]: " << cloud_filtered->points.size() << " points"
             << " (" << time_ms << " ms)" << endl;

        pcl::io::savePCDFileBinary(output_dir + "/passthrough_z.pcd", *cloud_filtered);
    }

    // Filter on X axis
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-0.5, 0.5);

        pass.filter(*cloud_filtered);
        cout << "  X filter [-0.5, 0.5]: " << cloud_filtered->points.size() << " points" << endl;

        pcl::io::savePCDFileBinary(output_dir + "/passthrough_x.pcd", *cloud_filtered);
    }

    // Negative filter (remove points in range)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZ>);

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-0.5, 0.5);
        pass.setNegative(true);  // Remove points in range

        pass.filter(*cloud_filtered);
        cout << "  Y filter (outside [-0.5, 0.5]): " << cloud_filtered->points.size() << " points" << endl;

        pcl::io::savePCDFileBinary(output_dir + "/passthrough_y_neg.pcd", *cloud_filtered);
    }

    cout << endl;
}

void demoFilterPipeline(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                         const string& output_dir) {
    cout << "=== Filter Pipeline (SLAM Pre-processing) ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    auto cloud_current = cloud;

    // Step 1: Range filtering (PassThrough)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_range(
        new pcl::PointCloud<pcl::PointXYZ>);
    {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_current);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-5.0, 5.0);
        pass.filter(*cloud_range);
        cout << "  1. Range filter: " << cloud_range->points.size() << " points" << endl;
        cloud_current = cloud_range;
    }

    // Step 2: Downsampling (VoxelGrid)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(
        new pcl::PointCloud<pcl::PointXYZ>);
    {
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud_current);
        voxel.setLeafSize(0.1f, 0.1f, 0.1f);
        voxel.filter(*cloud_downsampled);
        cout << "  2. Voxel grid (0.1m): " << cloud_downsampled->points.size() << " points" << endl;
        cloud_current = cloud_downsampled;
    }

    // Step 3: Outlier removal (SOR)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clean(
        new pcl::PointCloud<pcl::PointXYZ>);
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_current);
        sor.setMeanK(30);
        sor.setStddevMulThresh(1.0);
        sor.filter(*cloud_clean);
        cout << "  3. SOR: " << cloud_clean->points.size() << " points" << endl;
    }

    float total_reduction = 100.0f * (1.0f - static_cast<float>(cloud_clean->points.size()) /
                                             static_cast<float>(cloud->points.size()));
    cout << "  Total reduction: " << total_reduction << "%" << endl;

    pcl::io::savePCDFileBinary(output_dir + "/pipeline_output.pcd", *cloud_clean);

    cout << endl;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    cout << "=== PCL Filtering Tutorial ===" << endl;
    cout << "Point cloud filtering and preprocessing\n" << endl;

    // Determine output directory
    string output_dir = ".";
    if (argc > 1) {
        output_dir = argv[1];
    }
    cout << "Output directory: " << output_dir << endl << endl;

    // Generate noisy synthetic cloud
    cout << "Generating noisy point cloud..." << endl;
    auto cloud = generateNoisyCloud(10000, 0.1f);
    pcl::io::savePCDFileBinary(output_dir + "/noisy_input.pcd", *cloud);
    cout << endl;

    // Run filter demonstrations
    demoVoxelGrid(cloud, output_dir);
    demoStatisticalOutlierRemoval(cloud, output_dir);
    demoRadiusOutlierRemoval(cloud, output_dir);
    demoPassThrough(cloud, output_dir);
    demoFilterPipeline(cloud, output_dir);

    cout << "=== Filtering Demo Complete ===" << endl;
    return 0;
}
