# Measuring Leakage in Concept-Based Methods: An Information-Theoretic Approach

This repository contains the code for the paper _"Measuring Leakage in Concept-Based Methods: An Information-Theoretic Approach."_ The paper introduces an information-theoretic metric to quantify leakage in Concept Bottleneck Models (CBMs). This repository includes the metric's implementation and simulation experiments to validate its effectiveness. Additionally, it provides an initial implementation for comparing the estimated leakage across various concept-based methods, including classical CBMs (soft and hard configurations, trained sequentially, jointly, and independently), autoregressive CBMs, and embedding-based CBMs. Note however that this repository primarily focuses on validating the proposed leakage measure, with an emphasis on soft classical CBMs (joint training setting).

## TL;DR: Theoretical Background

Concept Bottleneck Models (CBMs) enhance interpretability by mapping input features \(x\) to human-understandable concepts \(\hat{c} = h(x)\), which are then used to predict the output \(\hat{y} = g(\hat{c})\). Ideally, \(g\) should remain simple to ensure transparency. However, CBMs are susceptible to **information leakage**, where unintended information beyond the defined concepts influences predictions. This leakage compromises interpretability by allowing predictions to depend on extra, undefined information.

Quantifying leakage is crucial for assessing interpretability violations. This research proposes an **information-theoretic measure** for leakage and validates it using synthetic data with controlled leakage. CBMs are trained on this data to evaluate the measure's reliability, with future work aimed at extending the study to real-world datasets.

## Code

- **`estimate_leakage_concept_methods.py`** – Implements the proposed leakage measure and validation experiments.
- **`validate_leakage_measure.py`** – Provides an initial (non-modularized) implementation for estimating leakage across different concept-based methods.
