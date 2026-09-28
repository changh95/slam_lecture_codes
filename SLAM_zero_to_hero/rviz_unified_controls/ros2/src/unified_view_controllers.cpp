// One mouse scheme for every RViz2 demo in the course:
//   left drag   = rotate
//   wheel       = zoom
//   right drag  = pan      (stock RViz2: zoom)
//   middle drag = pan      (unchanged)
//
// RViz2 has no mouse-binding setting, so each controller below is the stock
// one with the right button relabelled as the middle button before the event
// is handed to the stock handleMouseEvent. Everything else (properties, saved
// config keys, focal-point marker, wheel) is the upstream code path.

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/viewport_mouse_event.hpp>
#include <rviz_default_plugins/view_controllers/follower/third_person_follower_view_controller.hpp>
#include <rviz_default_plugins/view_controllers/orbit/orbit_view_controller.hpp>
#include <rviz_default_plugins/view_controllers/ortho/fixed_orientation_ortho_view_controller.hpp>

namespace slam_zero_to_hero
{
namespace vc = rviz_default_plugins::view_controllers;

inline void rightDragToPan(rviz_common::ViewportMouseEvent & event)
{
  if (event.buttons_down & Qt::RightButton) {
    event.buttons_down &= ~Qt::MouseButtons(Qt::RightButton);
    event.buttons_down |= Qt::MiddleButton;
  }
  if (event.acting_button == Qt::RightButton) {
    event.acting_button = Qt::MiddleButton;
  }
}

template<class Base>
class Unified : public Base
{
public:
  void handleMouseEvent(rviz_common::ViewportMouseEvent & event) override
  {
    rightDragToPan(event);
    Base::handleMouseEvent(event);
    // The stock handler just wrote its own button help into the status bar.
    this->setStatus(
      "<b>Left-Drag:</b> Rotate.  <b>Right-Drag / Middle-Drag:</b> Pan.  "
      "<b>Wheel:</b> Zoom.");
  }
};

class UnifiedOrbit : public Unified<vc::OrbitViewController> {};
class UnifiedThirdPersonFollower : public Unified<vc::ThirdPersonFollowerViewController> {};
class UnifiedTopDownOrtho : public Unified<vc::FixedOrientationOrthoViewController> {};

}  // namespace slam_zero_to_hero

PLUGINLIB_EXPORT_CLASS(slam_zero_to_hero::UnifiedOrbit, rviz_common::ViewController)
PLUGINLIB_EXPORT_CLASS(slam_zero_to_hero::UnifiedThirdPersonFollower, rviz_common::ViewController)
PLUGINLIB_EXPORT_CLASS(slam_zero_to_hero::UnifiedTopDownOrtho, rviz_common::ViewController)
