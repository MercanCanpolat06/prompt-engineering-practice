Experiment 1: Zero-Shot vs. One-Shot Performance on qwen2.5:14b
Date: 2026-02-28
Hypothesis: Adding a detailed One-Shot example will improve the JSON output formatting and reasoning accuracy compared to Zero-Shot.

Changes: simplifed the prompt.
Picked a seed to make the experiment reliable, and retriable.

Results & Observations:
Zero-Shot Accuracy: 4/13
One-Shot Accuracy: 5/13 (Performance increased, but no important difference)

observed that model does not understand constraints like "between" in the questions. I will look into that.

