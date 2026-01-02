# 3.17 Kimera-RPGO (Robust Pose Graph Optimization)

이 실습에서는 Kimera-RPGO를 사용한 Robust Pose Graph Optimization을 다룹니다.

## Kimera-RPGO란?

**Kimera-RPGO**는 MIT에서 개발한 Robust Pose Graph Optimization 라이브러리입니다. Loop closure의 outlier를 자동으로 검출하고 제거하여 강인한 SLAM 백엔드를 구현합니다.

### 주요 특징

- **Pairwise Consistency Maximization (PCM)**: Outlier loop closure 검출
- **Graduated Non-Convexity (GNC)**: Robust 최적화
- **GTSAM 기반**: Factor graph 최적화
- **Incremental/Batch 모드**: 실시간 및 오프라인 지원

## 파일 구성

```
3_17/
├── README.md              # 이 파일
├── CMakeLists.txt         # 빌드 설정
├── Dockerfile             # Docker 환경
└── examples/
    ├── rpgo_basics.cpp              # 기본 RPGO 사용법
    └── rpgo_outlier_rejection.cpp   # Outlier rejection 예제
```

## 핵심 개념

### 1. Robust Pose Graph Optimization이 필요한 이유

```
정상적인 Loop Closure:        잘못된 Loop Closure (Outlier):
    ○───○                         ○───○
    │   │                         │   ╲
    │   │                         │    X (잘못된 연결)
    ○───○                         ○───┘

Outlier가 있으면 전체 맵이 왜곡됩니다!
```

### 2. PCM (Pairwise Consistency Maximization)

Loop closure 후보들 간의 **일관성**을 검사합니다:

```
Loop Closure 1: x₁ → x₅
Loop Closure 2: x₂ → x₆

두 loop closure가 일관적인가?
→ Odometry와 함께 확인

Consistent: 포함
Inconsistent: 제거
```

### 3. GNC (Graduated Non-Convexity)

점진적으로 robust kernel을 적용:

```
μ = ∞  → Convex (L2 loss)
μ ↓    → 점점 더 robust
μ = 0  → Non-convex (Truncated loss)
```

## 빌드 방법

### Docker 사용 (권장)

```bash
# 이미지 빌드
docker build -t slam-3-17 .

# 컨테이너 실행
docker run -it slam-3-17

# 예제 실행
./build/rpgo_basics
./build/rpgo_outlier_rejection
```

### 로컬 빌드

**의존성:**
- GTSAM >= 4.0
- Kimera-RPGO

```bash
mkdir build && cd build
cmake ..
make -j4
```

## 예제 코드

### 기본 사용법

```cpp
#include <KimeraRPGO/RobustSolver.h>
#include <KimeraRPGO/SolverParams.h>

// Solver 설정
KimeraRPGO::RobustSolverParams params;
params.setPcmSimple3DParams(
    translation_threshold,
    rotation_threshold,
    KimeraRPGO::Verbosity::QUIET);

// Solver 생성
auto solver = KimeraRPGO::RobustSolver(params);

// Factor 추가
solver.update(new_factors, new_values);

// 결과 얻기
gtsam::Values result = solver.calculateEstimate();
```

### Outlier Rejection

```cpp
// Loop closure factor 추가
gtsam::BetweenFactor<gtsam::Pose3> loop_factor(
    from_key, to_key, relative_pose, noise_model);

solver.update({loop_factor}, {});

// Outlier인지 확인
if (solver.isLoopClosureRejected(from_key, to_key)) {
    std::cout << "Loop closure rejected as outlier!" << std::endl;
}
```

## 실습 과제

1. **기본 RPGO 사용**: `rpgo_basics.cpp` 실행
2. **Outlier 주입 실험**: 의도적으로 잘못된 loop closure 추가
3. **PCM 파라미터 조정**: threshold 변경에 따른 영향 분석
4. **GNC vs PCM 비교**: 두 방법의 차이 분석

## 참고 자료

- [Kimera-RPGO GitHub](https://github.com/MIT-SPARK/Kimera-RPGO)
- [Kimera Paper](https://arxiv.org/abs/1910.02490)
- [PCM Paper](https://ieeexplore.ieee.org/document/8460187)
- [GNC Paper](https://arxiv.org/abs/1909.08605)

## 다음 단계

이 실습을 마친 후:
- 3.18: SLAM 시스템 전체 아키텍처
- 실제 SLAM 시스템에 RPGO 적용
