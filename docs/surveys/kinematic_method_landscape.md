# Kinematic Method Landscape

This repo starts with a survey rather than model code. The point is to decide
what the sandbox should benchmark before building loaders, training loops, or
leaderboard machinery.

## Problem framing

The working scope is multivariate time-series classification over motion-derived
signals such as:

- position and displacement
- velocity and heading
- acceleration and jerk
- IMU channels
- track residuals and motion-model likelihoods

That covers several related domains:

- human activity recognition from inertial data
- vehicle maneuver classification
- target or trajectory mode classification from tracks

## Traditional methods

These are the first baseline family to keep:

- sliding-window summary statistics with logistic regression, SVM, random
  forest, or gradient boosting
- frequency-domain and autocorrelation features over fixed windows
- HMM-style sequence models for explicit motion-state transitions

Why they matter:

- they work well on modest datasets
- they are interpretable
- they provide the calibration point for any deeper model

## Model-based methods

For a genuinely kinematic sandbox, the classical tracking branch should be a
first-class citizen rather than an afterthought.

Core methods:

- constant-velocity and constant-acceleration model banks
- coordinated-turn, CTRV, and CTRA motion models
- residual-based maneuver classification
- IMM or MMAE multiple-model inference
- Bayesian class-matched filter banks with class-conditioned constraints
- open-set unknown handling with broad fallback dynamics

Why they matter:

- they encode physics more directly
- they are strong online baselines
- they provide a different failure mode than purely learned classifiers

## Recommended tracking-classification baseline

The repo should frame the first serious model-based benchmark as Bayesian joint
tracking and classification with class-conditioned physical constraints.

Recommended structure:

1. Class-matched filter bank across object classes.
2. IMM-style maneuver switching inside each class.
3. Soft envelope likelihoods from covariance, not hard threshold gates.
4. Optional ballistic-coefficient and `L/D` evidence when observable.
5. Explicit unknown-class mass so unsupported behavior is not forced into a
   known class.

That baseline is closer to operational kinematic classification than a plain
feature classifier over speed, turn rate, and altitude summaries.

## Deep learned methods

These should come after the classical baselines are stable.

- 1D CNN or TCN on raw windows as the first deep benchmark
- CNN-RNN hybrids when local structure and sequence order both matter
- transformer classifiers for longer context or multimodal sensor fusion
- self-supervised pretraining when labels are scarce

## Recommended benchmark order

1. Feature-engineered windows plus simple classifiers
2. Multiple-model kinematic baselines
3. 1D CNN or TCN deep baseline
4. Transformer and self-supervised extensions

## First repo implication

The first implementation pass should focus on:

- survey notes
- a method catalog
- artifact generation for the survey summary
- benchmark planning interfaces
- an implementation plan for Bayesian joint tracking and classification

It should not jump straight to training code before the benchmark families and
evaluation assumptions are stable.
