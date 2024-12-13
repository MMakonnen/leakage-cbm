# Measuring Leakage in Concept-Based Methods

This repository contains an initial proposal for an information-theoretic metric to quantify leakage in Concept Bottleneck Models (CBMs). It includes the metric's implementation and ongoing tests to validate its effectiveness. **This is an ongoing research project** at the Medical Data Science Group, ETH Zurich.

For details on the proposal, theoretical motivation, and context, see the `theory` folder.

### TL;DR: Theoretical Background

Concept Bottleneck Models (CBMs) enhance interpretability by mapping input features \(x\) to human-understandable concepts \(\hat{c} = h(x)\), which are then used to predict the output \(\hat{y} = g(\hat{c})\). Ideally, the function \(g\) is simple to maintain transparency. However, CBMs can suffer from **information leakage**, where unintended information beyond the specified concepts influences the final prediction. This leakage reduces interpretability because predictions can depend on extra information outside the defined concepts.

Quantifying this leakage is crucial to assess the extent of interpretability violations. This research investigates an **information-theoretic measure** for quantifying leakage. We validate this measure by constructing synthetic data with controlled leakage and testing its performance. Next, the CBM is trained on this data to evaluate the measure's effectiveness, with plans to apply it to real-world data for further validation.

### Next Steps

- Run simulations on synthetic data to check if the proposed measure behaves as expected.
- Compare the measure’s behavior on synthetic data when training CBMs on subsets of the synthetic data. Ensure its behavior aligns with results from fully synthetic experiments.
- Test the measure on real-world data.
