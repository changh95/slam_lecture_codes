#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

void colorize(const pcl::PointCloud<pcl::PointXYZ> &pc,
              pcl::PointCloud<pcl::PointXYZRGB> &pc_colored,
              const std::vector<int> &color) {

    int N              = pc.points.size();

    pc_colored.clear();
    pcl::PointXYZRGB pt_tmp;
    for (int         i = 0; i < N; ++i) {
        const auto &pt = pc.points[i];
        pt_tmp.x = pt.x;
        pt_tmp.y = pt.y;
        pt_tmp.z = pt.z;
        pt_tmp.r = color[0];
        pt_tmp.g = color[1];
        pt_tmp.b = color[2];
        pc_colored.points.emplace_back(pt_tmp);
    }
}

int main (int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "./pcd_viewer [path_to_pcd_file]" << std::endl;
    return -1;
  }

  const std::string filename = argv[1];

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (filename, *cloud) == -1)
  {
    const std::string error_message = "Couldn't read file " + filename;
    std::cerr << error_message << std::endl;
    return -1;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_c(new pcl::PointCloud<pcl::PointXYZRGB>);
  colorize(*cloud, *cloud_c, {255, 0, 0});

  pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
  viewer1.addPointCloud<pcl::PointXYZRGB>(cloud_c, "cloud");

  while (!viewer1.wasStopped()) {
    viewer1.spin();
  }

  return 0;
}
