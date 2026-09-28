// One mouse scheme for every RViz (ROS 1) demo in the course:
//   left drag   = rotate
//   wheel       = zoom
//   right drag  = pan      (stock RViz: zoom)
//   middle drag = pan      (unchanged)
//
// RViz has no mouse-binding setting, so each controller below is the stock one
// with the right button relabelled as the middle button before the event is
// handed to the stock handleMouseEvent. Everything else (properties, saved
// config keys, focal-point marker, wheel) is the upstream code path.

#include <pluginlib/class_list_macros.hpp>
#include <rviz/default_plugin/view_controllers/fixed_orientation_ortho_view_controller.h>
#include <rviz/default_plugin/view_controllers/orbit_view_controller.h>
#include <rviz/default_plugin/view_controllers/third_person_follower_view_controller.h>
#include <rviz/viewport_mouse_event.h>

namespace slam_zero_to_hero
{
inline void rightDragToPan(rviz::ViewportMouseEvent& event)
{
  if (event.buttons_down & Qt::RightButton)
  {
    event.buttons_down &= ~Qt::MouseButtons(Qt::RightButton);
    event.buttons_down |= Qt::MiddleButton;
  }
  if (event.acting_button == Qt::RightButton)
    event.acting_button = Qt::MiddleButton;
}

template <class Base>
class Unified : public Base
{
public:
  void handleMouseEvent(rviz::ViewportMouseEvent& event) override
  {
    rightDragToPan(event);
    Base::handleMouseEvent(event);
    // The stock handler just wrote its own button help into the status bar.
    this->setStatus("<b>Left-Drag:</b> Rotate.  <b>Right-Drag / Middle-Drag:</b> Pan.  "
                    "<b>Wheel:</b> Zoom.");
  }
};

class UnifiedOrbit : public Unified<rviz::OrbitViewController>
{
};
class UnifiedThirdPersonFollower : public Unified<rviz::ThirdPersonFollowerViewController>
{
};
class UnifiedTopDownOrtho : public Unified<rviz::FixedOrientationOrthoViewController>
{
};

}  // namespace slam_zero_to_hero

PLUGINLIB_EXPORT_CLASS(slam_zero_to_hero::UnifiedOrbit, rviz::ViewController)
PLUGINLIB_EXPORT_CLASS(slam_zero_to_hero::UnifiedThirdPersonFollower, rviz::ViewController)
PLUGINLIB_EXPORT_CLASS(slam_zero_to_hero::UnifiedTopDownOrtho, rviz::ViewController)
