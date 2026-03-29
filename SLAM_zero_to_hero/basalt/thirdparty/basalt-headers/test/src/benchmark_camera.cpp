#include <benchmark/benchmark.h>

#include <basalt/camera/generic_camera.hpp>

template <class CamT>
void bmProject(benchmark::State &state) {
  static constexpr int SIZE = 50;

  using Vec4 = typename CamT::Vec4;
  using Vec2 = typename CamT::Vec2;

  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  Vec4 p(0, 0, 5, 1);

  for (auto _ : state) {
    for (const CamT &cam : test_cams) {
      for (int x = -SIZE; x < SIZE; x++) {
        for (int y = -SIZE; y < SIZE; y++) {
          p[0] = x;
          p[1] = y;

          Vec2 res;
          benchmark::DoNotOptimize(cam.project(p, res));
        }
      }
    }
  }
}

template <class CamT>
void bmProjectJacobians(benchmark::State &state) {
  static constexpr int SIZE = 50;

  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  using Mat24 = typename CamT::Mat24;
  using Mat2N = typename CamT::Mat2N;

  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  Mat24 J_p;
  Mat2N J_param;

  Vec4 p(0, 0, 5, 1);

  for (auto _ : state) {
    for (const CamT &cam : test_cams) {
      for (int x = -SIZE; x <= SIZE; x++) {
        for (int y = -SIZE; y <= SIZE; y++) {
          p[0] = x;
          p[1] = y;

          Vec2 res;
          benchmark::DoNotOptimize(cam.project(p, res, &J_p, &J_param));
        }
      }
    }
  }
}

template <class CamT>
void bmUnproject(benchmark::State &state) {
  static constexpr int SIZE = 50;

  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  for (auto _ : state) {
    for (const CamT &cam : test_cams) {
      Vec2 p_center(cam.getParam()(2), cam.getParam()(3));

      for (int x = -SIZE; x <= SIZE; x++) {
        for (int y = -SIZE; y <= SIZE; y++) {
          Vec2 p = p_center;
          p[0] += x;
          p[1] += y;

          Vec4 res;
          benchmark::DoNotOptimize(cam.unproject(p, res));
        }
      }
    }
  }
}

template <class CamT>
void bmUnprojectJacobians(benchmark::State &state) {
  static constexpr int SIZE = 50;

  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  using Mat42 = typename CamT::Mat42;
  using Mat4N = typename CamT::Mat4N;

  Mat42 J_p;
  Mat4N J_param;

  for (auto _ : state) {
    for (const CamT &cam : test_cams) {
      Vec2 p_center(cam.getParam()(2), cam.getParam()(3));

      for (int x = -SIZE; x <= SIZE; x++) {
        for (int y = -SIZE; y <= SIZE; y++) {
          Vec2 p = p_center;
          p[0] += x;
          p[1] += y;

          Vec4 res;

          benchmark::DoNotOptimize(cam.unproject(p, res, &J_p, &J_param));
        }
      }
    }
  }
}

BENCHMARK_TEMPLATE(bmProject, basalt::PinholeCamera<double>);
BENCHMARK_TEMPLATE(bmProject, basalt::PinholeRadtan8Camera<double>);
BENCHMARK_TEMPLATE(bmProject, basalt::ExtendedUnifiedCamera<double>);
BENCHMARK_TEMPLATE(bmProject, basalt::UnifiedCamera<double>);
BENCHMARK_TEMPLATE(bmProject, basalt::KannalaBrandtCamera4<double>);
BENCHMARK_TEMPLATE(bmProject, basalt::DoubleSphereCamera<double>);
BENCHMARK_TEMPLATE(bmProject, basalt::FovCamera<double>);

BENCHMARK_TEMPLATE(bmProjectJacobians, basalt::PinholeCamera<double>);
BENCHMARK_TEMPLATE(bmProjectJacobians, basalt::PinholeRadtan8Camera<double>);
BENCHMARK_TEMPLATE(bmProjectJacobians, basalt::ExtendedUnifiedCamera<double>);
BENCHMARK_TEMPLATE(bmProjectJacobians, basalt::UnifiedCamera<double>);
BENCHMARK_TEMPLATE(bmProjectJacobians, basalt::KannalaBrandtCamera4<double>);
BENCHMARK_TEMPLATE(bmProjectJacobians, basalt::DoubleSphereCamera<double>);
BENCHMARK_TEMPLATE(bmProjectJacobians, basalt::FovCamera<double>);

BENCHMARK_TEMPLATE(bmUnproject, basalt::PinholeCamera<double>);
BENCHMARK_TEMPLATE(bmUnproject, basalt::PinholeRadtan8Camera<double>);
BENCHMARK_TEMPLATE(bmUnproject, basalt::ExtendedUnifiedCamera<double>);
BENCHMARK_TEMPLATE(bmUnproject, basalt::UnifiedCamera<double>);
BENCHMARK_TEMPLATE(bmUnproject, basalt::KannalaBrandtCamera4<double>);
BENCHMARK_TEMPLATE(bmUnproject, basalt::DoubleSphereCamera<double>);
BENCHMARK_TEMPLATE(bmUnproject, basalt::FovCamera<double>);

BENCHMARK_TEMPLATE(bmUnprojectJacobians, basalt::PinholeCamera<double>);
// BENCHMARK_TEMPL(bmUnprojectJacobians, basalt::PinholeRadtan8Camera<double>);
BENCHMARK_TEMPLATE(bmUnprojectJacobians, basalt::ExtendedUnifiedCamera<double>);
BENCHMARK_TEMPLATE(bmUnprojectJacobians, basalt::UnifiedCamera<double>);
BENCHMARK_TEMPLATE(bmUnprojectJacobians, basalt::KannalaBrandtCamera4<double>);
BENCHMARK_TEMPLATE(bmUnprojectJacobians, basalt::DoubleSphereCamera<double>);
BENCHMARK_TEMPLATE(bmUnprojectJacobians, basalt::FovCamera<double>);

BENCHMARK_MAIN();
