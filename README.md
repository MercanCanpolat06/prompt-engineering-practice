# Prompt Engineering Practice

This project is divided into 4 categories of prompt engineering practices.  
In each category, Qwen 2.5 14b model is tested against various constraints and prompts.

### Categories:
1. **Algorithmic Reasoning & Constraint Satisfaction** <br>
I have translated verbal logic questions of the ALES exam (from Türkiye) to use them as a question set. Since these questions contain strict constraints and multiple variables, they are a good fit to test the model against.
Zero-Shot prompt and a chain of thoughts prompt are compared. <br>
This is a *pilot study* with a small data set. <br>
Without CoT prompting the model scored 4 / 13. <br>
With CoT prompting the score raised to 7 / 13. (%75 improvement)

2. **In-Context Learning**

3. **Information Extraction** <br>
From a customer service data set, I compared various prompts to extract data.
I simplified the data set by only using elements that contained the word "order" in its ground truth. <br>
Next steps of this experiment should be done in a cleaned and larger data set, but since the goal of this repository is to learn different methods, it is enough.
Some problems I encountered were: 
- The model not deciding on what information is extracting worthy
- The model extracting the correct information, but in inconsistent format.

My prompts were promptA (zero-shot), promptB (one-shot) and promptC (few-shots).
One-shot significantly improved the score, in comparison to one shot, as it taught the model the output format. However there was no big jump between one-shot and few-shot. The extra computation load might be unnecessary.<br>

#### Results: <br>
Saved in the results.md file, in the information extraction folder.

## Acknowledgements / Credits

This project utilizes the **[jonathansuru/customer_service_information_extraction]** dataset, which is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

* **Original Creator/Author:** [jonathansuru]
* **Source:** [https://huggingface.co/datasets/jonathansuru/customer_service_information_extraction)]