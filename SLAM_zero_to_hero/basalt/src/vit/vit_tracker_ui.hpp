/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2022-2024, Collabora Ltd.
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

@file
@brief UI for Basalt when providing the VIT interface
@author Mateo de Mayo <mateo.demayo@collabora.com>
@author Simon Zeni <simon.zeni@collabora.com>
*/

#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <magic_enum.hpp>

#include <CLI/CLI.hpp>

#include <sophus/se3.hpp>

#include <pangolin/display/display.h>
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/pangolin.h>

#include <basalt/io/marg_data_io.h>
#include <basalt/optical_flow/optical_flow.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/keypoints.h>
#include <basalt/utils/vio_config.h>
#include <basalt/utils/vis_matrices.h>
#include <basalt/utils/vis_utils.h>
#include <basalt/vi_estimator/vio_estimator.h>

#define ASSERT(cond, ...)                                      \
  do { /* NOLINT(cppcoreguidelines-avoid-do-while) */          \
    if (!(cond)) {                                             \
      printf("Assertion failed @%s:%d\n", __func__, __LINE__); \
      printf(__VA_ARGS__);                                     \
      printf("\n");                                            \
      exit(EXIT_FAILURE); /* NOLINT(concurrency-mt-unsafe) */  \
    }                                                          \
  } while (false);
#define ASSERT_(cond) ASSERT(cond, "%s", #cond);

namespace basalt::vit_implementation {

using namespace Eigen;
using pangolin::Var;
using pangolin::View;
using std::cout;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::thread;
using std::vector;
using vis::UIMAT;

class vit_tracker_ui : public vis::VIOUIBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  VioVisualizationData::Ptr curr_vis_data = nullptr;
  VioVisualizationData::Ptr prev_vis_data = nullptr;
  pangolin::DataLog vio_data_log;
  thread vis_thread;
  thread ui_runner_thread;
  size_t num_cams = 0;
  std::atomic<bool> running = false;

 public:
  tbb::concurrent_bounded_queue<VioVisualizationData::Ptr> out_vis_queue{};
  tbb::concurrent_queue<double> *opt_flow_depth_queue = nullptr;

  VioVisualizationData::Ptr get_curr_vis_data() override { return curr_vis_data; }

  void initialize(int ncams) {
    vio_data_log.Clear();
    ASSERT_(ncams > 0);
    num_cams = ncams;
  }

  void start(const Sophus::SE3d &T_w_i_init, const Calibration<double> &calib, const VioConfig &config,
             const OpticalFlowBase::Ptr &of, const VioEstimatorBase::Ptr &ve) {
    opt_flow_depth_queue = &of->input_depth_queue;
    opt_flow = of;
    vio = ve;
    running = true;
    start_visualization_thread();
    start_ui(T_w_i_init, calib, config);
  }

  void stop() {
    running = false;
    vis_thread.join();
    ui_runner_thread.join();
  }

  void start_visualization_thread() {
    vis_thread = thread([&]() {
      while (true) {
        auto curr = curr_vis_data;
        out_vis_queue.pop(curr_vis_data);
        show_frame = show_frame + 1;
        if (curr_vis_data.get() == nullptr) break;
        prev_vis_data = curr;
      }
      cout << "Finished vis_thread\n";
    });
  }

  int64_t start_t_ns = -1;
  std::vector<int64_t> vio_t_ns;
  Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;

  UIMAT get_mat_to_show() { return show_blocks ? (UIMAT)mat_to_show.Get() : UIMAT::NONE; }

  void log_vio_data(const PoseVelBiasState<double>::Ptr &data) {
    int64_t t_ns = data->t_ns;
    if (start_t_ns < 0) start_t_ns = t_ns;
    float since_start_ns = t_ns - start_t_ns;

    if (data->input_images->state_reset) {
      vio_t_ns.clear();
      vio_t_w_i.clear();
      vio_data_log.Clear();
    }

    vio_t_ns.emplace_back(t_ns);
    vio_t_w_i.emplace_back(data->T_w_i.translation());

    Sophus::SE3d T_w_i = data->T_w_i;
    Eigen::Vector3d vel_w_i = data->vel_w_i;
    Eigen::Vector3d bg = data->bias_gyro;
    Eigen::Vector3d ba = data->bias_accel;

    vector<float> vals;
    vals.push_back(since_start_ns * 1e-9);
    for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
    for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
    for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
    for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

    vio_data_log.Log(vals);
  }

  std::shared_ptr<pangolin::Plotter> plotter;
  pangolin::DataLog imu_data_log{};
  void start_ui(const Sophus::SE3d &T_w_i_init, const Calibration<double> &cal, const VioConfig &conf) {
    ui_runner_thread = thread(&vit_tracker_ui::ui_runner, this, T_w_i_init, cal, conf);
  }

  void ui_runner(const Sophus::SE3d &T_w_i_init, const Calibration<double> &cal, const VioConfig &conf) {
    calib = cal;
    config = conf;
    string window_name = "Basalt";
    pangolin::CreateWindowAndBind(window_name, 1800, 1000, vis::default_win_params);

    glEnable(GL_DEPTH_TEST);

    img_view_display = &pangolin::CreateDisplay();
    img_view_display->SetBounds(0.4, 1.0, UI_WIDTH, 0.4);
    img_view_display->SetLayout(pangolin::LayoutEqual);

    plotter = std::make_shared<pangolin::Plotter>(&imu_data_log, 0.0, 100, -10.0, 10.0, 0.01, 0.01);
    plotter->SetBackgroundColour(vis::C_BLUEGREY_DARK());
    plotter->SetAxisColour(vis::C_BLUEGREY_LIGHT());
    plotter->SetTickColour(vis::C_BLUEGREY());
    plot_display = &pangolin::CreateDisplay();
    plot_display->SetBounds(0.0, 0.4, UI_WIDTH, 1.0);
    plot_display->AddDisplay(*plotter);

    blocks_view = std::make_shared<pangolin::ImageView>();
    blocks_view->UseNN() = true;  // Disable antialiasing, can be toggled with N key
    blocks_view->extern_draw_function = [this](View & /*v*/) { draw_blocks_overlay(); };
    const int DEFAULT_W = 480;
    blocks_display = &pangolin::CreateDisplay();
    blocks_display->SetBounds(0.0, 0.6, UI_WIDTH, pangolin::Attach::Pix(UI_WIDTH_PIX + DEFAULT_W));
    blocks_display->AddDisplay(*blocks_view);
    blocks_display->Show(show_blocks);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, UI_WIDTH);
    for (size_t i = 0; i < menus.size(); i++) {
      pangolin::CreatePanel(menus_str[i]).SetBounds(0.0, 0.4, 0.0, UI_WIDTH);
      pangolin::Display(menus_str[i]).Show(*menus[i]);
    }

    while (img_view.size() < calib.intrinsics.size()) {
      auto iv = make_shared<vis::VIOImageView>(*this);
      iv->UseNN() = true;  // Disable antialiasing (toggle it back with the N key)

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display->AddDisplay(*iv);
      iv->extern_draw_function = [idx, this](View &v) {
        draw_image_overlay(dynamic_cast<pangolin::ImageView &>(v), idx);
      };
    }

    Eigen::Vector3d cam_p(0.5, -2, -2);
    cam_p = T_w_i_init.so3() * calib.T_i_c[0].so3() * cam_p;
    cam_p[2] = 1;

    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0, pangolin::AxisZ));

    View &display3D = pangolin::CreateDisplay();
    display3D.SetAspect(-640 / 480.0);
    display3D.SetBounds(0.4, 1.0, 0.4, 1.0);
    display3D.SetHandler(new pangolin::Handler3D(camera));

    while (running && !pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (follow) {
        // TODO: There is a small race condition here over
        // curr_vis_data that is also present in the original basalt examples
        if (curr_vis_data) {
          auto T_w_i = curr_vis_data->states.rbegin()->second;
          T_w_i.so3() = Sophus::SO3d();

          camera.Follow(T_w_i.matrix());
        }
      }

      display3D.Activate(camera);
      glClearColor(1.0, 1.0, 1.0, 1.0);

      draw_scene();

      img_view_display->Activate();

      if (fixed_depth.GuiChanged() && opt_flow_depth_queue != nullptr) {
        opt_flow_depth_queue->push(fixed_depth);
      }
      depth_guess = opt_flow->depth_guess;

      pangolin::GlPixFormat fmt;
      fmt.glformat = GL_LUMINANCE;
      fmt.gltype = GL_UNSIGNED_SHORT;
      fmt.scalable_internal_format = GL_LUMINANCE16;

      if (curr_vis_data && curr_vis_data->opt_flow_res && curr_vis_data->opt_flow_res->input_images) {
        auto img_data = curr_vis_data->opt_flow_res->input_images->img_data;

        for (size_t cam_id = 0; cam_id < num_cams; cam_id++) {
          if (img_data[cam_id].img) {
            img_view[cam_id]->SetImage(img_data[cam_id].img->ptr, img_data[cam_id].img->w, img_data[cam_id].img->h,
                                       img_data[cam_id].img->pitch, fmt);
          }
        }
        if (follow_highlight) do_follow_highlight(true, false);
      }

      if (highlight_landmarks.GuiChanged() || filter_highlights.GuiChanged() || show_highlights.GuiChanged()) {
        highlights = vis::parse_selection(highlight_landmarks);
        filter_highlights = filter_highlights && !highlights.empty();
        if (show_blocks) do_show_blocks();
      }

      if (mat_to_show.GuiChanged()) {
        mat_name = std::string(magic_enum::enum_name((UIMAT)mat_to_show.Get()));
        if (show_blocks) do_show_blocks();
      }

      if (show_blocks) do_show_blocks();

      if (follow_highlight.GuiChanged()) {
        follow_highlight = follow_highlight && !highlights.empty();
        do_follow_highlight(follow_highlight, true);
      }

      Var<bool> *selected_menu = nullptr;
      for (Var<bool> *menu : menus) {
        if (menu->GuiChanged()) {
          selected_menu = menu;
          break;
        }
      }
      if (selected_menu != nullptr) {
        for (size_t i = 0; i < menus.size(); i++) {
          *menus[i] = menus[i] != selected_menu ? false : *selected_menu;
          pangolin::Display(menus_str[i]).Show(*menus[i]);
        }
      }

      draw_plots();

      pangolin::FinishFrame();
    }

    pangolin::DestroyWindow(window_name);
    cout << "Finished ui_runner\n";
  }

  void draw_image_overlay(pangolin::ImageView &v, size_t cam_id) {
    UNUSED(v);
    if (curr_vis_data == nullptr) return;

    if (show_obs) do_show_obs(cam_id);
    if (show_flow) do_show_flow(cam_id);
    if (show_highlights) do_show_highlights(cam_id);
    if (show_tracking_guess) do_show_tracking_guess(cam_id, show_frame, prev_vis_data);
    if (show_matching_guess) do_show_matching_guesses(cam_id);
    if (show_recall_guess) do_show_recall_guesses(cam_id);
    if (show_masks) do_show_masks(cam_id);
    if (show_cam0_proj) do_show_cam0_proj(cam_id, depth_guess);
    if (show_grid) do_show_grid();
    if (show_safe_radius) do_show_safe_radius();
  }

  void draw_scene() {
    glPointSize(3);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor3ubv(cam_color);
    Eigen::aligned_vector<Eigen::Vector3d> sub_gt(vio_t_w_i.begin(), vio_t_w_i.end());
    pangolin::glDrawLineStrip(sub_gt);

    pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);

    if (!curr_vis_data) return;

    for (size_t i = 0; i < calib.T_i_c.size(); i++) {
      if (!curr_vis_data->states.empty()) {
        const auto &[ts, p] = *curr_vis_data->states.rbegin();
        do_render_camera(p * calib.T_i_c[i], i, ts, cam_color);
      } else if (!curr_vis_data->frames.empty()) {
        const auto &[ts, p] = *curr_vis_data->frames.rbegin();
        do_render_camera(p * calib.T_i_c[i], i, ts, cam_color);
      }
    }

    for (const auto &[ts, p] : curr_vis_data->states)
      for (size_t i = 0; i < calib.T_i_c.size(); i++) do_render_camera(p * calib.T_i_c[i], i, ts, state_color);

    for (const auto &[ts, p] : curr_vis_data->frames)
      for (size_t i = 0; i < calib.T_i_c.size(); i++) do_render_camera(p * calib.T_i_c[i], i, ts, pose_color);

    for (const auto &[ts, p] : curr_vis_data->ltframes)
      for (size_t i = 0; i < calib.T_i_c.size(); i++) do_render_camera(p * calib.T_i_c[i], i, ts, vis::BLUE);

    glColor3ubv(pose_color);
    if (!filter_highlights) pangolin::glDrawPoints(curr_vis_data->points);

    Eigen::aligned_vector<Eigen::Vector3d> highlighted_points;
    if (show_highlights || filter_highlights) {
      for (size_t i = 0; i < curr_vis_data->point_ids.size(); i++) {
        Vector3d pos = curr_vis_data->points.at(i);
        int id = curr_vis_data->point_ids.at(i);
        if (is_highlighted(id)) highlighted_points.push_back(pos);
      }
    }

    if (filter_highlights) pangolin::glDrawPoints(highlighted_points);

    if (show_highlights) {
      glColor3ubv(vis::GREEN);
      glPointSize(10);
      pangolin::glDrawPoints(highlighted_points);
    }

    glColor3ubv(pose_color);
    if (show_ids) {
      for (size_t i = 0; i < curr_vis_data->points.size(); i++) {
        Vector3d pos = curr_vis_data->points.at(i);
        int id = curr_vis_data->point_ids.at(i);

        bool highlighted = is_highlighted(id);
        if (filter_highlights && !highlighted) continue;

        if (show_highlights && highlighted) glColor3ubv(vis::GREEN);
        FONT.Text("%d", id).Draw(pos.x(), pos.y(), pos.z());
        if (show_highlights && highlighted) glColor3ubv(pose_color);
      }
    }
  }

  OpticalFlowInput::Ptr last_img_data{};
  void update_last_image(OpticalFlowInput::Ptr &data) { last_img_data = data; }
  void draw_plots() {
    plotter->ClearSeries();
    plotter->ClearMarkers();

    if (show_est_pos) {
      plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine, pangolin::Colour::Red(), "position x", &vio_data_log);
      plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine, pangolin::Colour::Green(), "position y", &vio_data_log);
      plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "position z", &vio_data_log);
    }

    if (show_est_vel) {
      plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine, pangolin::Colour::Red(), "velocity x", &vio_data_log);
      plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine, pangolin::Colour::Green(), "velocity y", &vio_data_log);
      plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "velocity z", &vio_data_log);
    }

    if (show_est_bg) {
      plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine, pangolin::Colour::Red(), "gyro bias x", &vio_data_log);
      plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine, pangolin::Colour::Green(), "gyro bias y",
                         &vio_data_log);
      plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "gyro bias z", &vio_data_log);
    }

    if (show_est_ba) {
      plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine, pangolin::Colour::Red(), "accel bias x",
                         &vio_data_log);
      plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine, pangolin::Colour::Green(), "accel bias y",
                         &vio_data_log);
      plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine, pangolin::Colour::Blue(), "accel bias z",
                         &vio_data_log);
    }

    if (last_img_data) {
      double t = last_img_data->t_ns * 1e-9;
      plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal, pangolin::Colour::White());
    }
  }
};
}  // namespace basalt::vit_implementation
