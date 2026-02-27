Experiment 1: Zero-Shot vs. One-Shot Performance on Llama-3.1-8B
Date: 2026-02-27
Hypothesis: Adding a detailed One-Shot example will improve the JSON output formatting and reasoning accuracy compared to Zero-Shot.

Results & Observations:
Zero-Shot Accuracy: 6/13
One-Shot Accuracy: 2/13 (Performance degraded)
Analysis (Why did it fail?):
Prompt Overfitting: The model fixated on the specific entities (Ali, Berrak, Crepes) in the One-Shot example instead of generalizing the logic.
Format Confusion: Forcing the model to write a long reasoning chain inside a JSON value broke its output structure.
Next Steps / Fixes:
change the One-Shot prompt.
Move the reasoning process outside the JSON object using <thinking> tags.
I will deploy the modal locally to be able to make changes quicker.
