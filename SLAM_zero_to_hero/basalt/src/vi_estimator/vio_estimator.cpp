/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <basalt/vi_estimator/vio_estimator.h>

#include <basalt/vi_estimator/sqrt_keypoint_vio.h>
#include <basalt/vi_estimator/sqrt_keypoint_vo.h>

namespace basalt {

namespace {

template <class Scalar>
VioEstimatorBase::Ptr factory_helper(const VioConfig& config, const Calibration<double>& cam, const Eigen::Vector3d& g,
                                     bool use_imu) {
  VioEstimatorBase::Ptr res;

  if (use_imu) {
    res.reset(new SqrtKeypointVioEstimator<Scalar>(g, cam, config));

  } else {
    res.reset(new SqrtKeypointVoEstimator<Scalar>(cam, config));
  }

  return res;
}

}  // namespace

VioEstimatorBase::Ptr VioEstimatorFactory::getVioEstimator(const VioConfig& config, const Calibration<double>& cam,
                                                           const Eigen::Vector3d& g, bool use_imu, bool use_double) {
  if (use_double) {
#ifdef BASALT_INSTANTIATIONS_DOUBLE
    return factory_helper<double>(config, cam, g, use_imu);
#else
    BASALT_LOG_FATAL("Compiled without double support.");
#endif
  } else {
#ifdef BASALT_INSTANTIATIONS_FLOAT
    return factory_helper<float>(config, cam, g, use_imu);
#else
    BASALT_LOG_FATAL("Compiled without float support.");
#endif
  }
}

double alignSVD(const std::vector<int64_t>& filter_t_ns, const Eigen::aligned_vector<Eigen::Vector3d>& filter_t_w_i,
                const std::vector<int64_t>& gt_t_ns, Eigen::aligned_vector<Eigen::Vector3d>& gt_t_w_i) {
  Eigen::aligned_vector<Eigen::Vector3d> est_associations;
  Eigen::aligned_vector<Eigen::Vector3d> gt_associations;

  for (size_t i = 0; i < filter_t_w_i.size(); i++) {
    int64_t t_ns = filter_t_ns[i];

    size_t j;
    for (j = 0; j < gt_t_ns.size(); j++) {
      if (gt_t_ns.at(j) > t_ns) break;
    }
    j--;

    if (j >= gt_t_ns.size() - 1) {
      continue;
    }

    double dt_ns = t_ns - gt_t_ns.at(j);
    double int_t_ns = gt_t_ns.at(j + 1) - gt_t_ns.at(j);

    BASALT_ASSERT_STREAM(dt_ns >= 0, "dt_ns " << dt_ns);
    BASALT_ASSERT_STREAM(int_t_ns > 0, "int_t_ns " << int_t_ns);

    // Skip if the interval between gt larger than 100ms
    if (int_t_ns > 1.1e8) continue;

    double ratio = dt_ns / int_t_ns;

    BASALT_ASSERT(ratio >= 0);
    BASALT_ASSERT(ratio < 1);

    Eigen::Vector3d gt = (1 - ratio) * gt_t_w_i[j] + ratio * gt_t_w_i[j + 1];

    gt_associations.emplace_back(gt);
    est_associations.emplace_back(filter_t_w_i[i]);
  }

  int num_kfs = est_associations.size();

  Eigen::Matrix<double, 3, Eigen::Dynamic> gt, est;
  gt.setZero(3, num_kfs);
  est.setZero(3, num_kfs);

  for (size_t i = 0; i < est_associations.size(); i++) {
    gt.col(i) = gt_associations[i];
    est.col(i) = est_associations[i];
  }

  Eigen::Vector3d mean_gt = gt.rowwise().mean();
  Eigen::Vector3d mean_est = est.rowwise().mean();

  gt.colwise() -= mean_gt;
  est.colwise() -= mean_est;

  Eigen::Matrix3d cov = gt * est.transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) S(2, 2) = -1;

  Eigen::Matrix3d rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
  Eigen::Vector3d trans = mean_gt - rot_gt_est * mean_est;

  Sophus::SE3d T_gt_est(rot_gt_est, trans);
  Sophus::SE3d T_est_gt = T_gt_est.inverse();

  for (auto& i : gt_t_w_i) i = T_est_gt * i;

  double error = 0;
  for (size_t i = 0; i < est_associations.size(); i++) {
    est_associations[i] = T_gt_est * est_associations[i];
    Eigen::Vector3d res = est_associations[i] - gt_associations[i];

    error += res.transpose() * res;
  }

  error /= est_associations.size();
  error = std::sqrt(error);

  std::cout << "T_align\n" << T_gt_est.matrix() << std::endl;
  std::cout << "error " << error << std::endl;
  std::cout << "number of associations " << num_kfs << std::endl;

  return error;
}

int associate(const std::vector<int64_t>& filter_t_ns, const Eigen::aligned_vector<Eigen::Vector3d>& filter_t_w_i,
              const std::vector<int64_t>& gt_t_ns, const Eigen::aligned_vector<Eigen::Vector3d>& gt_t_w_i,  //
              Eigen::Matrix<int64_t, Eigen::Dynamic, 1>& out_ts, Eigen::Matrix<float, 3, Eigen::Dynamic>& out_est_xyz,
              Eigen::Matrix<float, 3, Eigen::Dynamic>& out_ref_xyz) {
  // BASALT_ASSERT(filter_t_ns.size() == filter_t_w_i.size() && gt_t_ns.size() == gt_t_ns.size());
  int num_est = filter_t_ns.size();
  int num_ref = gt_t_ns.size();
  // BASALT_ASSERT(num_est < num_ref);

  out_ref_xyz.resize(3, num_est);
  out_est_xyz.resize(3, num_est);
  out_ts.resize(num_est);

  int num_assocs = 0;

  Eigen::Index i = 0;  // est index
  Eigen::Index j = 0;  // ref index

  // Advance est to first ref or after
  while (filter_t_ns[i] < gt_t_ns[0] && i < num_est) i++;

  for (; i < num_est; i++) {
    int64_t t_ns = filter_t_ns[i];

    // j is -1
    for (; j < num_ref; j++) {
      if (gt_t_ns[j] > t_ns) break;
    }
    j--;  // j will never be -1 because i starts such that est(i) > ref(0)

    if (j >= num_ref - 1) continue;

    double dt_ns = t_ns - gt_t_ns[j];
    double int_t_ns = gt_t_ns[j + 1] - gt_t_ns[j];
    // BASALT_ASSERT(dt_ns >= 0 && int_t_ns > 0);

    if (int_t_ns > 1.1e8) continue;  // Skip if >100ms

    double ratio = dt_ns / int_t_ns;
    // BASALT_ASSERT(ratio >= 0 && ratio < 1);

    Eigen::Vector3f gt = (1 - ratio) * gt_t_w_i[j].cast<float>() + ratio * gt_t_w_i[j + 1].cast<float>();
    out_ref_xyz.col(num_assocs) = gt;
    out_est_xyz.col(num_assocs) = filter_t_w_i[i].cast<float>();
    out_ts(num_assocs) = t_ns;
    num_assocs++;
  }

  out_ref_xyz.conservativeResize(Eigen::NoChange, num_assocs);
  out_est_xyz.conservativeResize(Eigen::NoChange, num_assocs);
  out_ts.conservativeResize(num_assocs);
  return num_assocs;
}

Eigen::Matrix4f get_alignment(const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& est_xyz,
                              const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& ref_xyz,  //
                              int i, int j) {
  // BASALT_ASSERT(est_xyz.cols() == ref_xyz.cols());
  // int pose_count = est_xyz.cols();
  // BASALT_ASSERT(i < j && i >= 0 && i < pose_count && j >= 0 && j <= pose_count);

  // Get block i-j without copy
  auto estb_xyz = est_xyz.block(0, i, 3, j - i);
  auto refb_xyz = ref_xyz.block(0, i, 3, j - i);

  Eigen::Vector3f mean_est = estb_xyz.rowwise().mean();
  Eigen::Vector3f mean_ref = refb_xyz.rowwise().mean();

  Eigen::Matrix<float, 3, Eigen::Dynamic> c_est = estb_xyz.colwise() - mean_est;
  Eigen::Matrix<float, 3, Eigen::Dynamic> c_ref = refb_xyz.colwise() - mean_ref;
  Eigen::Matrix3f cov = c_ref * c_est.transpose();
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3f S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) S(2, 2) = -1;

  Eigen::Matrix3f rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
  Eigen::Vector3f trans = mean_ref - rot_gt_est * mean_est;

  Sophus::SE3f T_ref_est(rot_gt_est, trans);

  return T_ref_est.matrix();
}

float compute_ate(const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& est_xyz,
                  const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>>& ref_xyz,  //
                  const Eigen::Ref<Eigen::Matrix4f>& T_ref_est_mat,                          //
                  int i, int j) {
  // BASALT_ASSERT(est_xyz.cols() == ref_xyz.cols());
  int pose_count = est_xyz.cols();
  // BASALT_ASSERT(i < j && i >= 0 && i < pose_count && j >= 0 && j <= pose_count);

  Sophus::SE3f T_ref_est(T_ref_est_mat);

  float rmse = 0;

  for (Eigen::Index k = i; k < j; k++) {
    Eigen::Vector3f res = T_ref_est * est_xyz.col(k) - ref_xyz.col(k);
    rmse += res.transpose() * res;
  }

  rmse = std::sqrt(rmse / pose_count);

  return rmse;
}

}  // namespace basalt
