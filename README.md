# Measuring Leakage in Concept-Based Methods

This repository introduces an information-theoretic metric to quantify leakage in Concept Bottleneck Models (CBMs). It includes the metric's implementation and ongoing tests to validate its effectiveness. Additionally, it features an initial comparison of estimated leakage across various concept-based methods, including classical CBMs (soft and hard configurations, trained sequentially, jointly, and independently), autoregressive CBMs, and embedding-based CBMs. **This is an ongoing research project** at the Medical Data Science Group, ETH Zurich.

### TL;DR: Theoretical Background

Concept Bottleneck Models (CBMs) improve interpretability by mapping input features \(x\) to human-understandable concepts \(\hat{c} = h(x)\), which are then used to predict the output \(\hat{y} = g(\hat{c})\). Ideally, \(g\) remains simple to ensure transparency. However, CBMs are prone to **information leakage**, where unintended information beyond the specified concepts influences predictions. This leakage undermines interpretability by allowing predictions to depend on extra, undefined information.

Quantifying leakage is critical for assessing interpretability violations. This research proposes an **information-theoretic measure** for leakage and validates it using synthetic data with controlled leakage. The CBM is trained on this data to evaluate the measure's reliability, with future plans to extend the study to real-world datasets for further validation.
