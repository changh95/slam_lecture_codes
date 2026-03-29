[![pipeline status](https://gitlab.com/VladyslavUsenko/basalt-headers/badges/master/pipeline.svg)](https://gitlab.com/VladyslavUsenko/basalt-headers/commits/master)
[![coverage report](https://gitlab.com/VladyslavUsenko/basalt-headers/badges/master/coverage.svg)](https://gitlab.com/VladyslavUsenko/basalt-headers/commits/master)

## Basalt Headers
This repository contains reusable components of Basalt project as header-only library ([Documentation](https://vladyslavusenko.gitlab.io/basalt-headers/)). For more information see https://vision.in.tum.de/research/vslam/basalt.

This library includes:
* Camera models.
* Standalone image datatype with support for views, interpolation, gradients, and image pyramids.
* Uniform B-Splines for Rd (d-dimentional vectors), SO(3) and SE(3).
* Preintegration of inertial measurement unit (IMU) data.
* Data types to store IMU-camera calibration.
* Cereal serialization for basalt types as well as some Eigen and Sophus types.



## Related Publications
Implemented camera models:
* **The Double Sphere Camera Model**, V. Usenko and N. Demmel and D. Cremers, In 2018 International Conference on 3D Vision (3DV), [[DOI:10.1109/3DV.2018.00069]](https://doi.org/10.1109/3DV.2018.00069), [[arXiv:1807.08957]](https://arxiv.org/abs/1807.08957).

Implemented IMU preintegration:
* **Visual-Inertial Mapping with Non-Linear Factor Recovery**, V. Usenko, N. Demmel, D. Schubert, J. Stückler, D. Cremers, In IEEE Robotics and Automation Letters (RA-L) [[DOI:10.1109/LRA.2019.2961227]](https://doi.org/10.1109/LRA.2019.2961227), [[arXiv:1904.06504]](https://arxiv.org/abs/1904.06504).

B-spline trajectory representation:
* **Efficient Derivative Computation for Cumulative B-Splines on Lie Groups**, C. Sommer, V. Usenko, D. Schubert, N. Demmel, D. Cremers, In 2020 Conference on Computer Vision and Pattern Recognition (CVPR), [[DOI:10.1109/CVPR42600.2020.01116]](https://doi.org/10.1109/CVPR42600.2020.01116), [[arXiv:1911.08860]](https://arxiv.org/abs/1911.08860).




## Licence

The code is provided under a BSD 3-clause license. See the LICENSE file for details.
Note also the different licenses of thirdparty submodules.

`image.h` is adapted from [Pangolin](https://github.com/stevenlovegrove/Pangolin) by Steven Lovegrove (MIT license).
