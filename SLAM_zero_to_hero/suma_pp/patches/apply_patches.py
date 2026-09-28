#!/usr/bin/env python3
"""Patch semantic_suma (531954d) + rangenet_lib (3fc223e) for this image.

1. rangenet_lib: drop in the TensorRT 10 port of netTensorRT.{hpp,cpp}
   (the upstream code uses the TensorRT 5 API, removed in TensorRT 8/10,
   and TensorRT >= 10.8 is the first to support the RTX 50xx / sm_120).
2. semantic_suma/CMakeLists.txt: build without catkin, C++17 (TensorRT 10
   headers need it).
3. VisualizerWindow: an autorun mode driven by environment variables, so the
   GUI can play a sequence, write poses + runtime + screenshots and quit
   without anyone clicking the play button:
     SUMA_AUTOPLAY=1          start playing right after the scan is opened
     SUMA_OUTPUT_DIR=/results where poses/runtime/screenshots go
     SUMA_MAX_SCANS=N         stop after N scans (default: whole sequence)
     SUMA_SNAPSHOT_EVERY=K    also save the 3D view every K scans
     SUMA_EXIT=1              quit when done

Every replacement must match exactly once, otherwise the build fails.
"""
import shutil
import sys
from pathlib import Path

WS = Path(sys.argv[1] if len(sys.argv) > 1 else "/ws")
PATCHES = Path(__file__).resolve().parent


def replace(path, old, new, count=1):
    p = WS / path
    s = p.read_text()
    n = s.count(old)
    if n != count:
        sys.exit(f"patch failed: {path}: expected {count} match(es), found {n}:\n{old}")
    p.write_text(s.replace(old, new))


# 1. TensorRT 10 port -----------------------------------------------------------
shutil.copy(PATCHES / "rangenet_lib/netTensorRT.hpp", WS / "rangenet_lib/include/netTensorRT.hpp")
shutil.copy(PATCHES / "rangenet_lib/netTensorRT.cpp", WS / "rangenet_lib/src/netTensorRT.cpp")

# 2. catkin-free CMake -----------------------------------------------------------
replace("semantic_suma/CMakeLists.txt",
        "find_package(catkin COMPONENTS glow rangenet_lib)\n", "")
replace("semantic_suma/CMakeLists.txt", """catkin_package(
  INCLUDE_DIRS src
  LIBRARIES suma
  CATKIN_DEPENDS
    glow
    rangenet_lib
  DEPENDS
    Boost
)
""", "")
replace("semantic_suma/CMakeLists.txt", '"-std=c++11 -O3  -Wall', '"-std=c++17 -O3 -Wall -Wno-deprecated-declarations')
# as a sub-project CMAKE_SOURCE_DIR is the workspace; keep bin/ inside semantic_suma
replace("semantic_suma/CMakeLists.txt", "${CMAKE_SOURCE_DIR}/bin", "${PROJECT_SOURCE_DIR}/bin")

# 3. autorun mode -----------------------------------------------------------------
H = "semantic_suma/src/visualizer/VisualizerWindow.h"
C = "semantic_suma/src/visualizer/VisualizerWindow.cpp"

replace(H, "  void openFile(const QString& filename);",
        """  void openFile(const QString& filename);
  /** autorun: play, dump poses/runtime/screenshots, optionally quit (env vars, see apply_patches.py) **/
  void startAutorun();""")
replace(H, "  QTimer timer_;", """  QTimer timer_;
  bool autorun_{false};
  uint32_t autorunMaxScans_{0};
  uint32_t autorunSnapshotEvery_{0};
  std::string autorunOutput_{"."};
  void finishAutorun();
  void writePoses(const std::string& pose_file, const std::string& runtime_file);
  void saveView(const std::string& filename);""")

replace(C, "#include <QtWidgets/QFileDialog>\n",
        "#include <QtWidgets/QFileDialog>\n#include <QtCore/QDir>\n#include <QtCore/QTimer>\n"
        "#include <QtGui/QScreen>\n#include <QtWidgets/QApplication>\n#include <cstdlib>\n")

replace(C, "void VisualizerWindow::nextScan() {\n  if (reader_ == nullptr) return;\n",
        """void VisualizerWindow::nextScan() {
  if (reader_ == nullptr) return;

  if (autorun_ && (currentScanIdx_ >= reader_->count() - 1 ||
                   (autorunMaxScans_ > 0 && currentScanIdx_ + 1 >= autorunMaxScans_))) {
    finishAutorun();
    return;
  }
""")

replace(C, "  ui_.wCanvas->setOdomPoses(fusion_->getIntermediateOdometryPoses());\n",
        """  ui_.wCanvas->setOdomPoses(fusion_->getIntermediateOdometryPoses());

  if (autorun_ && currentScanIdx_ % 100 == 0) {
    SurfelMapping::Stats s = fusion_->getStatistics();
    std::cout << "[autorun] scan " << currentScanIdx_ << " / " << reader_->count()
              << "  complete-time " << s["complete-time"] * 1000.0 << " ms" << std::endl;
  }
  if (autorun_ && autorunSnapshotEvery_ > 0 && currentScanIdx_ % autorunSnapshotEvery_ == 0) {
    saveView(autorunOutput_ + "/frames/" +
             QString("%1.png").arg((int)currentScanIdx_, 5, 10, (QChar)'0').toStdString());
  }
""")

replace(C, "void VisualizerWindow::initializeGraph() {", r'''void VisualizerWindow::saveView(const std::string& filename) {
  ui_.wCanvas->updateGL();
  QImage img = ui_.wCanvas->grabFrameBuffer();
  if (img.save(QString::fromStdString(filename)))
    std::cout << "[autorun] wrote " << filename << " (" << img.width() << "x" << img.height() << ")" << std::endl;
  else
    std::cerr << "[autorun] could not write " << filename << std::endl;
}

void VisualizerWindow::writePoses(const std::string& pose_file, const std::string& runtime_file) {
  // same as savePoses(), without the file dialogs: optimized poses in the KITTI camera frame.
  Eigen::Matrix4f T_cam_velo = Eigen::Matrix4f::Identity();
  if (calib_.exists("Tr")) T_cam_velo = calib_["Tr"];
  Eigen::Matrix4f T_velo_cam = T_cam_velo.inverse();

  std::ofstream out(pose_file);
  auto poses = fusion_->getOptimizedPoses();
  for (uint32_t i = 0; i < poses.size(); ++i) {
    Eigen::Matrix4f pose = T_cam_velo * poses[i].cast<float>() * T_velo_cam;
    for (uint32_t r = 0; r < 3; ++r)
      for (uint32_t c = 0; c < 4; ++c) out << ((r == 0 && c == 0) ? "" : " ") << pose(r, c);
    out << std::endl;
  }
  out.close();
  std::cout << "[autorun] wrote " << poses.size() << " poses to " << pose_file << std::endl;

  std::ofstream rt(runtime_file);
  rt << "# scan initialization preprocessing icp loop mapping complete (seconds)" << std::endl;
  for (uint32_t i = 0; i < statistics.size(); ++i) {
    rt << i << " " << statistics[i]["initialize-time"] << " " << statistics[i]["preprocessing-time"] << " "
       << statistics[i]["icp-time"] << " " << statistics[i]["loop-time"] << " " << statistics[i]["mapping-time"]
       << " " << statistics[i]["complete-time"] << std::endl;
  }
  rt.close();
}

void VisualizerWindow::startAutorun() {
  auto env = [](const char* k) { const char* v = std::getenv(k); return v ? std::string(v) : std::string(); };
  if (env("SUMA_AUTOPLAY") != "1" || reader_ == nullptr) return;
  autorun_ = true;
  if (!env("SUMA_OUTPUT_DIR").empty()) autorunOutput_ = env("SUMA_OUTPUT_DIR");
  if (!env("SUMA_MAX_SCANS").empty()) autorunMaxScans_ = std::stoul(env("SUMA_MAX_SCANS"));
  if (!env("SUMA_SNAPSHOT_EVERY").empty()) autorunSnapshotEvery_ = std::stoul(env("SUMA_SNAPSHOT_EVERY"));
  if (autorunSnapshotEvery_ > 0) QDir().mkpath(QString::fromStdString(autorunOutput_ + "/frames"));
  std::cout << "[autorun] playing " << (autorunMaxScans_ ? autorunMaxScans_ : reader_->count())
            << " scans, output -> " << autorunOutput_ << std::endl;
  ui_.chkFastMode->setChecked(true);
  play(true);
}

void VisualizerWindow::finishAutorun() {
  play(false);
  autorun_ = false;
  std::cout << "[autorun] finished at scan " << currentScanIdx_ << std::endl;
  writePoses(autorunOutput_ + "/poses.txt", autorunOutput_ + "/runtime.txt");

  // 1) the view the GUI shows while running (camera follows the car)
  saveView(autorunOutput_ + "/suma_follow.png");
  // 2) bird's-eye view over the whole trajectory
  ui_.chkFollowPose->setChecked(false);
  ui_.chkBirdsEyeView->setChecked(true);
  saveView(autorunOutput_ + "/suma_birdseye.png");
  // 3) the whole window (3D view + Qt panels), read back from the X server
  QApplication::processEvents();
  QPixmap win = QGuiApplication::primaryScreen()->grabWindow(winId());
  if (win.save(QString::fromStdString(autorunOutput_ + "/suma_window.png")))
    std::cout << "[autorun] wrote " << autorunOutput_ << "/suma_window.png (" << win.width() << "x" << win.height() << ")" << std::endl;
  ui_.chkBirdsEyeView->setChecked(false);
  ui_.chkFollowPose->setChecked(true);

  const char* ex = std::getenv("SUMA_EXIT");
  if (ex && std::string(ex) == "1") {
    // give an external screenshot tool (scripts/run_autorun.sh) time to grab the window
    const char* d = std::getenv("SUMA_EXIT_DELAY_MS");
    int delay = d ? std::atoi(d) : 0;
    QTimer::singleShot(delay, qApp, SLOT(quit()));
  }
}

void VisualizerWindow::initializeGraph() {''')

replace("semantic_suma/src/visualizer/visualizer.cpp",
        "    window.openFile(QString(argv[2]));\n",
        "    window.openFile(QString(argv[2]));\n    window.startAutorun();\n")

print("patches applied")
