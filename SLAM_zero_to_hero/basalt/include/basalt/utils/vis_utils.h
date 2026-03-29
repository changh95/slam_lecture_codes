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
#pragma once

#include <Eigen/Dense>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/colour.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/gl/glfont.h>

#include <basalt/utils/vis_matrices.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <pangolin/var/var.h>
#include <pangolin/var/varvaluegeneric.h>
#include <basalt/utils/sophus_utils.hpp>
#include <string>
#include <tuple>

namespace basalt::vis {
extern pangolin::GlFont FONT;
}
using basalt::vis::FONT;

const uint8_t cam_color[3]{250, 0, 125};
const uint8_t state_color[3]{250, 0, 26};
const uint8_t pose_color[3]{0, 50, 255};
const uint8_t gt_color[3]{0, 171, 47};
constexpr float MIN_DEPTH_COLOR[3]{0.27, 0.79, 1};      // blue
constexpr float MAX_DEPTH_COLOR[3]{1, 0.1, 0.42};       // pink
constexpr uint8_t MIN_DEPTH_COLOR_UB[3]{69, 201, 255};  // blue
constexpr uint8_t MAX_DEPTH_COLOR_UB[3]{255, 26, 107};  // pink

const float HIGHLIGHT_RADIUS = 20;  // Radius of the circle around the highlighted point

inline void render_camera(const Eigen::Matrix4d& T_w_c, float lineWidth, const uint8_t* color, float sizeFactor,
                          bool show_ids = false, size_t frame_idx = 0, const uint8_t* idx_color = nullptr,
                          bool show_fwd = false) {
  const float sz = sizeFactor;
  const float width = 640, height = 480, fx = 500, fy = 500, cx = 320, cy = 240;

  Eigen::aligned_vector<Eigen::Vector3f> lines = {{0, 0, 0},
                                                  {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
                                                  {0, 0, 0},
                                                  {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
                                                  {0, 0, 0},
                                                  {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
                                                  {0, 0, 0},
                                                  {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
                                                  {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
                                                  {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
                                                  {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
                                                  {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
                                                  {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
                                                  {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
                                                  {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
                                                  {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz}};
  if (show_fwd) {
    lines.emplace_back(0, 0, 0);
    lines.emplace_back(0, 0, 1);
  }

  glPushMatrix();
  glMultMatrixd(T_w_c.data());
  glColor3ubv(color);
  glLineWidth(lineWidth);
  pangolin::glDrawLines(lines);
  if (show_ids) {
    glColor3ubv(idx_color);
    FONT.Text("%d", frame_idx).Draw(0, 0, -0.01F);
  }
  glPopMatrix();
}

inline void getcolor(float p, float np, float& r, float& g, float& b) {
  float inc = 4.0 / np;
  float x = p * inc;
  r = 0.0f;
  g = 0.0f;
  b = 0.0f;

  if ((0 <= x && x <= 1) || (5 <= x && x <= 6))
    r = 1.0f;
  else if (4 <= x && x <= 5)
    r = x - 4;
  else if (1 <= x && x <= 2)
    r = 1.0f - (x - 1);

  if (1 <= x && x <= 3)
    g = 1.0f;
  else if (0 <= x && x <= 1)
    g = x - 0;
  else if (3 <= x && x <= 4)
    g = 1.0f - (x - 3);

  if (3 <= x && x <= 5)
    b = 1.0f;
  else if (2 <= x && x <= 3)
    b = x - 2;
  else if (5 <= x && x <= 6)
    b = 1.0f - (x - 5);
}

inline std::tuple<float, float, float> color_lerp(float t,                               //
                                                  const float min[3] = MIN_DEPTH_COLOR,  //
                                                  const float max[3] = MAX_DEPTH_COLOR   //
) {
  return {min[0] + t * (max[0] - min[0]),  //
          min[1] + t * (max[1] - min[1]),  //
          min[2] + t * (max[2] - min[2])};
}

inline std::tuple<uint8_t, uint8_t, uint8_t> color_lerp_ub(float t,                                      //
                                                           const uint8_t minub[3] = MIN_DEPTH_COLOR_UB,  //
                                                           const uint8_t maxub[3] = MAX_DEPTH_COLOR_UB   //
) {
  float min[3] = {minub[0] / 255.0F, minub[1] / 255.0F, minub[2] / 255.0F};
  float max[3] = {maxub[0] / 255.0F, maxub[1] / 255.0F, maxub[2] / 255.0F};
  auto [r, g, b] = color_lerp(t, min, max);
  return {uint8_t(r * 255.0F), uint8_t(g * 255.0F), uint8_t(b * 255.0F)};
}

template <typename P, int N, class Allocator>
void glDrawCirclePerimeters(const std::vector<Eigen::Matrix<P, N, 1>, Allocator>& points, float radius = 5.0) {
  for (auto& p : points) {
    pangolin::glDrawCirclePerimeter((GLfloat)p(0), (GLfloat)p(1), (GLfloat)radius);
  }
}

namespace basalt::vis {

using Eigen::MatrixXf;
using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::Vector4d;
using pangolin::Colour;
using pangolin::ImageView;
using pangolin::META_FLAG_READONLY;
using pangolin::MouseButton;
using pangolin::Var;
using pangolin::View;
using std::set;
using std::shared_ptr;
using std::string;
using std::unordered_map;
using std::vector;
using Button = Var<std::function<void(void)>>;

extern pangolin::GlFont SMALL_FONT;
extern pangolin::Params default_win_params;

const uint8_t BLUE[4]{0x21, 0x96, 0xF3, 0xFF};
const uint8_t GREEN[4]{0x4C, 0xAF, 0x50, 0xFF};
const uint8_t LGREEN[4]{0x76, 0xFF, 0x03, 0xFF};
const uint8_t RED[4]{0xF4, 0x43, 0x36, 0xFF};
const uint8_t YELLOW[4]{0xFF, 0xFF, 0x00, 0xFF};

inline Colour C_BLUEGREY() { return {69 / 255.0, 90 / 255.0, 100 / 255.0}; }
inline Colour C_RED() { return {244 / 255.0, 67 / 255.0, 54 / 255.0}; }
inline Colour C_PINK() { return {233 / 255.0, 30 / 255.0, 99 / 255.0}; }
inline Colour C_BLUE() { return {33 / 255.0, 150 / 255.0, 243 / 255.0}; }
inline Colour C_GREEN() { return {76 / 255.0, 175 / 255.0, 80 / 255.0}; }
inline Colour C_YELLOW() { return {255 / 255.0, 152 / 255.0, 0 / 255.0}; }
inline Colour C_AMBER() { return {255 / 255.0, 193 / 255.0, 7 / 255.0}; }

inline Colour C_BLUEGREY_DARK() { return {38 / 255.0, 50 / 255.0, 56 / 255.0}; }
inline Colour C_RED_DARK() { return {183 / 255.0, 28 / 255.0, 28 / 255.0}; }
inline Colour C_PINK_DARK() { return {136 / 255.0, 14 / 255.0, 79 / 255.0}; }
inline Colour C_BLUE_DARK() { return {13 / 255.0, 71 / 255.0, 161 / 255.0}; }
inline Colour C_GREEN_DARK() { return {27 / 255.0, 94 / 255.0, 32 / 255.0}; }
inline Colour C_YELLOW_DARK() { return {245 / 255.0, 127 / 255.0, 23 / 255.0}; }
inline Colour C_AMBER_DARK() { return {255 / 255.0, 111 / 255.0, 0 / 255.0}; }

inline Colour C_BLUEGREY_LIGHT() { return {236 / 255.0, 239 / 255.0, 241 / 255.0}; }
inline Colour C_RED_LIGHT() { return {255 / 255.0, 235 / 255.0, 238 / 255.0}; }
inline Colour C_BLUE_LIGHT() { return {227 / 255.0, 242 / 255.0, 253 / 255.0}; }
inline Colour C_GREEN_LIGHT() { return {232 / 255.0, 245 / 255.0, 233 / 255.0}; }

struct SelectionNode {
  bool is_range;
  KeypointId a;
  KeypointId b;

  SelectionNode(bool is_range, KeypointId a, KeypointId b) : is_range(is_range), a(a), b(b) {}
  bool contains(KeypointId n) const { return is_range ? a <= n && n <= b : n == a; }
};
using Selection = std::vector<SelectionNode>;

//! Parse a set of numbers described in @p str. Example inputs: "1,3,5-10", "1000-2000,3,5-7"
Selection parse_selection(const std::string& str);

//! The reverse of parse_selection
string selection_to_string(const Selection& selection);

//! Return a new selection without @p kpids
Selection remove_from_selection(const Selection& selection, const std::set<KeypointId>& kpids);

//! Return the subset of kps that are in selection
Keypoints filter_kps_by_selection(const Selection& selection, const Keypoints& kps);

void get_rect_containing_kps(const Keypoints& kpids, float& l, float& r, float& t, float& b);

bool is_selected(const Selection& selection, size_t n);

struct VIOUIBase {
  static constexpr int UI_WIDTH_PIX = 200;
  const pangolin::Attach UI_WIDTH = pangolin::Attach::Pix(UI_WIDTH_PIX);

  View* img_view_display;
  View* plot_display;
  View* blocks_display;
  shared_ptr<ImageView> blocks_view;
  vector<shared_ptr<ImageView>> img_view;
  bool show_blocks = false;
  Selection highlights{};
  VioConfig config;
  Calibration<double> calib;
  OpticalFlowBase::Ptr opt_flow;
  VioEstimatorBase::Ptr vio;
  // TODO: Make vis_map into a queue that stores a range of frames, even in realtime mode
  unordered_map<int64_t, VioVisualizationData::Ptr> vis_map;

  Var<int> show_frame{"ui.show_frame", 0, META_FLAG_READONLY};

  Var<bool> features_menu{"ui.Features Menu", false, true};
  Var<string> features_menu_title{"features_menu.MENU", "Features Menu", META_FLAG_READONLY};
  Var<bool> show_ids{"features_menu.show_ids", false, true};
  Var<bool> show_flow{"features_menu.show_flow", false, true};
  Var<bool> show_responses{"features_menu.show_responses", false, true};
  Var<bool> show_tracking_guess{"features_menu.show_tracking_guess", false, true};
  Var<bool> show_matching_guess{"features_menu.show_matching_guess", false, true};
  Var<bool> show_recall_guess{"features_menu.show_recall_guess", false, true};
  Var<bool> show_obs{"features_menu.show_obs", true, true};
  Var<bool> show_depth{"features_menu.show_depth", false, true};

  Var<bool> highlights_menu{"ui.Highlights Menu", false, true};
  Var<string> highlights_menu_title{"highlights_menu.MENU", "Highlights Menu", META_FLAG_READONLY};
  Var<string> highlight_landmarks{"highlights_menu.Highlight", ""};
  Var<bool> filter_highlights{"highlights_menu.filter_highlights", false, true};
  Var<bool> show_highlights{"highlights_menu.show_highlights", false, true};
  Var<bool> follow_highlight{"highlights_menu.follow_highlight", false, true};
  Button highlight_frame_btn{"highlights_menu.highlight_frame", [this]() { highlight_frame(); }};
  Button clear_highlights_btn{"highlights_menu.clear_highlights", [this]() { clear_highlights(); }};

  Var<bool> blocks_menu{"ui.Block Menu", false, true};
  Var<string> blocks_menu_title{"blocks_menu.MENU", "Block Menu", META_FLAG_READONLY};
  Button toggle_blocks_btn{"blocks_menu.toggle_blocks", [this]() { toggle_blocks(); }};
  Var<string> mat_name{"blocks_menu.mat_name", "Jr", META_FLAG_READONLY};
  Var<int> mat_to_show{"blocks_menu.mat_to_show", (int)UIMAT::JR, (int)UIMAT::JR, (int)UIMAT::COUNT - 1};
  Var<bool> show_block_vals{"blocks_menu.show_block_vals", false, true};

  Var<bool> keyframe_menu{"ui.Keyframe Menu", false, true};
  Var<string> keyframe_menu_title{"keyframe_menu.MENU", "Keyframe Menu", META_FLAG_READONLY};
  Button take_ltkf_btn{"keyframe_menu.Take Keyframe", [this]() { take_ltkf(); }};

  Var<bool> image_menu{"ui.Image Menu", false, true};
  Var<string> image_menu_title{"image_menu.MENU", "Image Menu", META_FLAG_READONLY};
  Var<bool> show_grid{"image_menu.show_grid", false, true};
  Var<bool> show_safe_radius{"image_menu.show_safe_radius", false, true};
  Var<bool> show_cam0_proj{"image_menu.show_cam0_proj", false, true};
  Var<bool> show_masks{"image_menu.show_masks", false, true};

  Var<bool> guesses_menu{"ui.Guesses Menu", false, true};
  Var<string> guesses_menu_title{"guesses_menu.MENU", "Guesses Menu", META_FLAG_READONLY};
  Var<bool> show_guesses{"guesses_menu.Show matching guesses", false, true};
  Var<bool> show_same_pixel_guess{"guesses_menu.SAME_PIXEL", true, true};
  Var<bool> show_reproj_avg_depth_guess{"guesses_menu.REPROJ_AVG_DEPTH", true, true};
  Var<bool> show_reproj_fix_depth_guess{"guesses_menu.REPROJ_FIX_DEPTH", true, true};
  Var<double> fixed_depth{"guesses_menu.FIX_DEPTH", 2, 0, 3};
  Var<bool> show_active_guess{"guesses_menu.Active Guess", true, true};
  Var<double> depth_guess{"guesses_menu.depth_guess", 2, META_FLAG_READONLY};

  Var<bool> curves_menu{"ui.Curves Menu", false, true};
  Var<string> curves_menu_title{"curves_menu.MENU", "Curves Menu", META_FLAG_READONLY};
  Var<bool> show_est_pos{"curves_menu.show_est_pos", true, true};
  Var<bool> show_est_vel{"curves_menu.show_est_vel", false, true};
  Var<bool> show_est_bg{"curves_menu.show_est_bg", false, true};
  Var<bool> show_est_ba{"curves_menu.show_est_ba", false, true};

  Var<bool> follow{"ui.follow", true, true};
  Button reset_state_btn{"ui.Reset State", [this]() { reset_state(); }};

  vector<Var<bool>*> menus{&features_menu, &highlights_menu, &blocks_menu, &keyframe_menu,
                           &image_menu,    &guesses_menu,    &curves_menu};
  vector<string> menus_str{"features_menu", "highlights_menu", "blocks_menu", "keyframe_menu",
                           "image_menu",    "guesses_menu",    "curves_menu", "trajectory_menu"};

  virtual VioVisualizationData::Ptr get_curr_vis_data() = 0;

  KeypointId get_kpid_at(size_t cam_id, int x, int y, float radius = 10);
  bool is_highlighted(size_t lmid) const { return vis::is_selected(highlights, lmid); }
  bool highlight_frame();
  bool highlight_kps_in_rect(size_t cam_id, float l, float r, float t, float b);
  bool remove_highlights(const std::set<KeypointId>& kpids);
  void clear_highlights();
  bool toggle_blocks();
  bool take_ltkf();
  bool reset_state();
  void do_show_flow(size_t cam_id);
  void do_show_highlights(size_t cam_id);
  void do_show_tracking_guess(size_t cam_id, size_t frame_id, const VioVisualizationData::Ptr& prev_vis_data);
  void do_show_tracking_guess_vio(size_t cam_id, size_t frame_id, const VioDatasetPtr& vio_dataset,
                                  const std::unordered_map<int64_t, VioVisualizationData::Ptr>& vis_map);
  void do_show_recall_guesses(size_t cam_id);
  void do_show_matching_guesses(size_t cam_id);
  void do_show_masks(size_t cam_id);
  void do_show_cam0_proj(size_t cam_id, double depth_guess);
  void do_show_grid();
  void do_show_safe_radius();
  void do_show_guesses(size_t cam_id);
  void do_show_obs(size_t cam_id);
  void draw_blocks_overlay();
  void draw_jacobian_overlay(const UIJacobians& uij);
  void draw_hessian_overlay(const UIHessians& uih);
  bool do_toggle_blocks();
  void do_show_blocks();
  void do_show_hessian(UIHessians& uih);
  void do_show_jacobian(UIJacobians& uij);
  bool do_follow_highlight(bool follow, bool smooth_zoom);

  void do_render_camera(const Sophus::SE3d& T_w_c, size_t i, size_t ts, const uint8_t* color);
};

struct VIOImageView : ImageView {
  VIOUIBase& ui;
  VIOImageView(VIOUIBase& ui);
  void Mouse(View& view, MouseButton button, int x_screen, int y_screen, bool pressed, int button_state) override;
  void Keyboard(View& view, unsigned char key, int x, int y, bool pressed) override;
};

}  // namespace basalt::vis
