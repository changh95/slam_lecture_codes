/**
 * Interactive ICP viewer shared by the two registration demos.
 *
 * Both demos open a viewer and step their ICP one iteration per keystroke, so
 * the point-to-plane demo can drive two methods at once and show that it
 * converges in far fewer iterations than point-to-point on the same data.
 *
 * Included only by the registration demos: it pulls in PCL's visualization
 * module (and with it VTK), which the odometry demo has no use for.
 */

#pragma once

#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <memory>

#include <pcl/common/io.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>

#include "demo_common.hpp"

namespace demo {

using PointNT = pcl::PointNormal;
using CloudNT = pcl::PointCloud<PointNT>;

/**
 * @brief True when an X display looks available
 *
 * The demos always visualize, so a missing display is worth reporting clearly
 * rather than letting VTK abort with a backtrace.
 */
inline bool hasDisplay()
{
    const char* display = std::getenv("DISPLAY");
    return display != nullptr && display[0] != '\0';
}

inline void reportMissingDisplay()
{
    std::cerr << "\nNo X display found (DISPLAY is unset), so the viewer cannot open.\n"
                 "The registration results above are complete. To see the alignment,\n"
                 "run with the host display forwarded:\n\n"
                 "  docker run -it --rm -e DISPLAY=$DISPLAY \\\n"
                 "      -v /tmp/.X11-unix:/tmp/.X11-unix \\\n"
                 "      slam_zero_to_hero:part2_ch03_06 <demo>\n";
}

/**
 * @brief How far a transform is from the identity
 *
 * Used to decide that an ICP has stopped moving: once a single iteration
 * changes the pose by less than a small fraction of the model, the method has
 * converged and the iteration it happened on is the interesting number.
 */
struct Increment
{
    double rotation_deg;
    double translation_m;
};

inline Increment incrementSize(const Eigen::Matrix4f& T)
{
    Eigen::Quaterniond q(T.block<3, 3>(0, 0).cast<double>());
    q.normalize();
    return {2.0 * std::atan2(q.vec().norm(), std::abs(q.w())) * 180.0 / M_PI,
            T.block<3, 1>(0, 3).cast<double>().norm()};
}

/** @brief Result of advancing one ICP iteration */
struct StepOutcome
{
    bool moved;                  ///< false once the iteration no longer changes the pose
    double fitness;
    Eigen::Matrix4f cumulative;  ///< source -> target so far
};

/**
 * @brief One ICP method being stepped in the viewer
 *
 * `display` is the cloud the viewer renders; `advance` runs a single iteration,
 * writes the new positions into `display`, and reports what happened.
 */
struct Track
{
    std::string label;
    std::string id;
    int r, g, b;
    CloudT::Ptr display;
    std::function<StepOutcome()> advance;

    // Filled in by the driver
    int iterations = 0;
    int converged_at = -1;
    double fitness = 0.0;
    Eigen::Matrix4f cumulative = Eigen::Matrix4f::Identity();
};

/** @brief Converged when one iteration barely moves the pose */
inline bool hasSettled(const Eigen::Matrix4f& increment, double scale)
{
    const Increment inc = incrementSize(increment);
    return inc.rotation_deg < 1e-3 && inc.translation_m < scale * 1e-6;
}

/**
 * @brief Run every track to convergence without a viewer, counting iterations
 *
 * This is what makes the point-to-plane comparison say something: at a fixed
 * iteration budget both methods reach the same answer on clean data, so the
 * number worth printing is how many iterations each one needed.
 *
 * The call that detects settling did no work, so it is not counted - that keeps
 * this number identical to the one the viewer shows.
 * @return iterations that actually moved the pose, or `cap` if it never settled
 */
inline int countIterationsToConverge(const std::function<StepOutcome()>& advance, int cap = 100)
{
    for (int i = 0; i < cap; ++i)
    {
        if (!advance().moved)
        {
            return i;
        }
    }
    return cap;
}

/**
 * @brief Build a stepper that advances point-to-point ICP one iteration per call
 *
 * `display` is both the working cloud and what the viewer renders: each call
 * aligns it one iteration closer to the target and rewrites it in place.
 */
inline std::function<StepOutcome()> makePointToPointStep(const CloudT::Ptr& target,
                                                        const CloudT::Ptr& display,
                                                        double max_correspondence_distance,
                                                        double scale)
{
    auto cumulative = std::make_shared<Eigen::Matrix4f>(Eigen::Matrix4f::Identity());

    return [=]() -> StepOutcome {
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(display);
        icp.setInputTarget(target);
        icp.setMaximumIterations(1);
        icp.setTransformationEpsilon(1e-12);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);

        CloudT::Ptr aligned(new CloudT);
        icp.align(*aligned);

        if (!icp.hasConverged())
        {
            return {false, 0.0, *cumulative};
        }

        const Eigen::Matrix4f increment = icp.getFinalTransformation();
        if (hasSettled(increment, scale))
        {
            return {false, icp.getFitnessScore(), *cumulative};
        }

        *cumulative = increment * *cumulative;
        *display = *aligned;
        return {true, icp.getFitnessScore(), *cumulative};
    };
}

/**
 * @brief Build a stepper that advances point-to-plane ICP one iteration per call
 *
 * The normals travel with the working cloud, so they are transformed by ICP
 * rather than re-estimated every iteration. `display` carries the positions out
 * to the viewer.
 */
inline std::function<StepOutcome()> makePointToPlaneStep(const CloudNT::Ptr& target,
                                                        const CloudNT::Ptr& source,
                                                        const CloudT::Ptr& display,
                                                        double max_correspondence_distance,
                                                        double scale)
{
    auto cumulative = std::make_shared<Eigen::Matrix4f>(Eigen::Matrix4f::Identity());

    return [=]() -> StepOutcome {
        pcl::IterativeClosestPointWithNormals<PointNT, PointNT> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaximumIterations(1);
        icp.setTransformationEpsilon(1e-12);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);

        CloudNT::Ptr aligned(new CloudNT);
        icp.align(*aligned);

        if (!icp.hasConverged())
        {
            return {false, 0.0, *cumulative};
        }

        const Eigen::Matrix4f increment = icp.getFinalTransformation();
        if (hasSettled(increment, scale))
        {
            return {false, icp.getFitnessScore(), *cumulative};
        }

        *cumulative = increment * *cumulative;
        *source = *aligned;
        pcl::copyPointCloud(*aligned, *display);
        return {true, icp.getFitnessScore(), *cumulative};
    };
}

/**
 * @brief Point the camera at the origin from a distance proportional to the model
 *
 * The Stanford bunny is modelled +Y up, not +Z, so the up vector follows the
 * model rather than the usual robotics convention - otherwise the demo opens
 * looking at the bunny lying on its side.
 */
inline void setupCamera(pcl::visualization::PCLVisualizer& viewer, double scale)
{
    viewer.setCameraPosition(0.5 * scale, 0.35 * scale, 1.7 * scale,  // camera position
                             0.0, 0.0, 0.0,                           // look at the origin
                             0.0, 1.0, 0.0);                          // up is +Y
    viewer.setCameraClipDistances(0.01 * scale, 100.0 * scale);
}

/**
 * @brief Step one or more ICP methods interactively, one iteration per keystroke
 *
 * The target is drawn once in blue; each track gets its own colour and advances
 * together with the others, so their convergence rates can be compared directly.
 */
inline void runStepViewer(const std::string& title,
                          const CloudT::Ptr& target,
                          std::vector<Track>& tracks,
                          double scale)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer(title));

    viewer->setBackgroundColor(0.1, 0.1, 0.1);

    // Small: the model is centred on the origin, so long axes would run straight
    // through the cloud and read as part of it
    viewer->addCoordinateSystem(0.12 * scale, "coordinate");

    // Target (blue), static
    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_color(target, 0, 100, 255);
    viewer->addPointCloud<PointT>(target, target_color, "target");
    // Drawn larger than the moving clouds so it stays visible underneath them
    // once a method has converged onto it
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "target");

    for (const auto& t : tracks)
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> color(t.display, t.r, t.g, t.b);
        viewer->addPointCloud<PointT>(t.display, color, t.id);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, t.id);
    }

    setupCamera(*viewer, scale);

    // Any keystroke advances one iteration. 'q' is excluded because that is the
    // viewer's own quit key; the keys PCLVisualizer binds itself (r to reset the
    // camera, s/w for the render mode, g for the grid) still do their usual job
    // in addition to stepping.
    bool next_step = false;
    viewer->registerKeyboardCallback(
        [&next_step](const pcl::visualization::KeyboardEvent& event) {
            if (event.keyDown() && event.getKeySym() != "q")
            {
                next_step = true;
            }
        });

    // Legend and per-track status, redrawn after every step
    const auto refreshText = [&]() {
        int y = 20 + static_cast<int>(tracks.size()) * 20;

        viewer->removeShape("legend");
        viewer->addText("Blue: target | press any key to step, 'q' to quit",
                        10, y + 20, 14, 1.0, 1.0, 1.0, "legend");

        for (const auto& t : tracks)
        {
            std::ostringstream ss;
            ss << t.label << ": " << t.iterations << " iter";
            if (t.converged_at > 0)
            {
                ss << "  CONVERGED at " << t.converged_at;
            }
            else
            {
                ss << "  fitness " << std::scientific << std::setprecision(2) << t.fitness;
            }

            viewer->removeShape("status_" + t.id);
            viewer->addText(ss.str(), 10, y, 14,
                            t.r / 255.0, t.g / 255.0, t.b / 255.0, "status_" + t.id);
            y -= 20;
        }
    };

    refreshText();

    std::cout << "\n=== Interactive ICP ===\n";
    std::cout << "Press any key in the viewer window to run one iteration.\n";
    std::cout << "Press 'q' to quit.\n\n";

    // The viewer stays alive after every track has settled, so the final
    // alignment can still be inspected.
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);

        if (next_step)
        {
            next_step = false;

            for (auto& t : tracks)
            {
                if (t.converged_at > 0)
                {
                    continue;
                }

                const StepOutcome outcome = t.advance();
                t.fitness = outcome.fitness;
                t.cumulative = outcome.cumulative;

                if (outcome.moved)
                {
                    ++t.iterations;
                }
                else
                {
                    t.converged_at = t.iterations;
                    std::cout << t.label << " converged after " << t.iterations
                              << " iterations\n";
                    continue;
                }

                std::cout << t.label << " iteration " << t.iterations
                          << " - fitness " << std::scientific << std::setprecision(4)
                          << t.fitness << "\n";
            }

            for (const auto& t : tracks)
            {
                pcl::visualization::PointCloudColorHandlerCustom<PointT>
                    color(t.display, t.r, t.g, t.b);
                viewer->updatePointCloud<PointT>(t.display, color, t.id);
            }
            refreshText();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

}  // namespace demo
