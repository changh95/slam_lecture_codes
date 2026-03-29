This document is a declaration of software quality for the `libconsole-bridge-dev` ROS external dependency, based on the guidelines in [REP-2004](https://www.ros.org/reps/rep-2004.html).

# libconsole-bridge-dev Quality Declaration

The ROS external dependency `libconsole-bridge-dev` claims to be in the **Quality Level 1** category.

Below are the rationales, notes, and caveats for this claim, organized by each requirement listed in the [Package Quality Categories in REP-2004](https://index.ros.org/doc/ros2/Contributing/Developer-Guide/#package-quality-categories) of the ROS2 developer guide.

## Version Policy [1]

### Version Scheme [1.i]

`libconsole-bridge-dev` uses `semver` according to the recommendation for ROS Core packages in the [ROS 2 Developer Guide](https://index.ros.org/doc/ros2/Contributing/Developer-Guide/#versioning).

### Version Stability [1.ii]

`libconsole-bridge-dev` is at a stable version, i.e. `>= 1.0.0`.

### Public API Declaration [1.iii]

All symbols in the installed headers are considered part of the public API.

### API Stability Policy [1.iv]

`libconsole-bridge-dev` is used as an upstream package within the ROS2 ecosystem and has been API/ABI stable for several years. If a breaking change is introduced, ROS2 distributions will pin to a specific major version.

### ABI Stability Policy [1.v]

`libconsole-bridge-dev` is used as an upstream package within the ROS2 ecosystem and has been API/ABI stable for several years. If a breaking change is introduced, ROS2 distributions will pin to a specific major version.

### ABI and ABI Stability Within a Released ROS Distribution [1.vi]

If a breaking change is introduced, ROS2 distributions will pin `libconsole-bridge-dev` to a specific major version.

## Change Control Process [2]

`libconsole-bridge-dev` follows the recommended guidelines for ROS Core packages in the [ROS 2 Developer Guide](https://index.ros.org/doc/ros2/Contributing/Developer-Guide/#change-control-process).

### Change Requests [2.i]

All changes will occur through a pull request, check [ROS 2 Developer Guide](https://index.ros.org/doc/ros2/Contributing/Developer-Guide/#change-control-process) for additional information.

### Contributor Origin [2.ii]

This package uses DCO as its confirmation of contributor origin policy. More info can be seen under the [contributing file](./CONTRIBUTING.md) of this repository.

### Peer Review Policy [2.iii]

All pull requests will be peer-reviewed, check [ROS 2 Developer Guide](https://index.ros.org/doc/ros2/Contributing/Developer-Guide/#change-control-process) for additional information.

### Continuous Integration [2.iv]

Pull requests must pass CI under Linux and Windows environments set in with AppVeyor and Travis-CI. Jobs are automatically trigered with each PR and the results shown in the Github repository.

Current test results are shown here:

[Linux (Travis CI)](https://travis-ci.org/github/ros/console_bridge)
[Windows (Appveyor)](https://ci.appveyor.com/project/tfoote/console-bridge)

###  Documentation Policy [2.v]

All pull requests must resolve related documentation changes before merging.

## Documentation [3]

### Feature Documentation [3.i]

`libconsole-bridge-dev` lists its features in the README file of its GitHub repository, [here](https://github.com/ros/console_bridge/#features). Also its [wiki](http://wiki.ros.org/console_bridge) provides additional documentation over its usage.

### Public API Documentation [3.ii]

All functions and classes in the public API of `libconsole-bridge-dev` include docblocks explaining their functionality or describing its usage.

### License [3.iii]

The license for `libconsole-bridge-dev` is 3-Clause BSD, and a summary is in each source file and a full copy of the license is in the [`LICENSE`](./LICENSE) file.

### Copyright Statements [3.iv]

The copyright holders each provide a statement of copyright in each source code file in `libconsole-bridge-dev`.

New source files added to this library will require having a copyright statement.

## Testing [4]

### Feature Testing [4.i]

`libconsole-bridge-dev` provides testing of its [features](https://github.com/ros/console_bridge/#features) under the [test folder](./test/).

### Public API Testing [4.ii]

`libconsole-bridge-dev` includes public API tests and new additions or changes to the public API require tests before being added.

The tests aim to cover both typical usage and corner cases, but are quantified by contributing to code coverage. Currently, the complete API of the package is not fully tested.

Current test results are shown here:

[Linux (Travis CI)](https://travis-ci.org/github/ros/console_bridge)
[Windows (Appveyor)](https://ci.appveyor.com/project/tfoote/console-bridge)

### Coverage [4.iii]

`libconsole-bridge-dev` provides coverage testing under its configured Travis-CI.

Current test results are shown here:

[Linux Coverage results(codecov)](https://codecov.io/gh/ros/console_bridge).

### Performance [4.iv]

The performance tests of this package are located in the [vendored library](https://github.com/ros2/console_bridge_vendor/tree/master/test/benchmark). The most recent test results can be found [here](http://build.ros2.org/view/Rci/job/Rci__benchmark_ubuntu_focal_amd64/BenchmarkTable/).

`libconsole-bridge-dev` doe s not provide performance testing.

### Linters and Static Analysis [4.v]

`libconsole-bridge-dev` is being tested with `cppcheck` and `cpplint`.

## Dependencies [5]

`libconsole-bridge-dev` has no run-time or build-time dependencies that need to be considered for this declaration.

## Platform Support [6]

`libconsole-bridge-dev` officially supports Ubuntu, Windows and MacOS systems. CI tests PRs with Ubuntu Trusty (Travis CI) and Windows Server 2019 (Appveyor).

## Security

### Vulnerability Disclosure Policy [7.i]

This package conforms to the Vulnerability Disclosure Policy in [REP-2006](https://www.ros.org/reps/rep-2006.html).
