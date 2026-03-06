# Prompt Engineering Practice

This project is divided into 4 categories of prompt engineering practices.  
In each category, Qwen 2.5 14b model is tested against various constraints and prompts.

### Categories:
**1. Algorithmic Reasoning & Constraint Satisfaction** <br>
I have translated verbal logic questions of the ALES exam (from Türkiye) to present them as a question set. Since these questions contain strict constraints and multiple variables, they are a good fit to test the model against.
Zero-Shot prompt and a chain of thoughts prompt are compared. <br>
This is a *pilot study* with a small data set. <br>
Without CoT prompting the model scored 4 / 13. <br>
With CoT prompting the score raised to 7 / 13. (%75 improvement)

**2. Dynamic Few-Shot Learning**<br>
This part compares standard few-shot prompting with dynamic few-shot prompting.
The data set is shuffelled and seperated into 2: pool and test. Prompt A is default few-shots prompt with static value. It consists of some examples I chose beforehand. Prompt B is dynamically created for each question, by comparing the question semantically with the vectors in the pool and choosing the closest 3. Test set consists of 20 questions.
#### Results:
By running [semantic_comparison.py](dynamic-few-shot/semantic_comparison.py), the results are printed to the terminal. Both answers are compared with the ground truth, and the closer one is marked.
Output for mine was as below: 
```
============================================================
DYNAMIC IN-CONTEXT LEARNING (SEMANTIC) RESULTS
============================================================
Scenario A (Static One-Shot) Average Score   : 68.23%
Scenario B (Dynamic Few-Shot) Average Score  : 70.87%
------------------------------------------------------------
Performance Improvement (Relative)           : + 3.86% better!
------------------------------------------------------------
Competition Statistics (Total 20 Questions):
   Dynamic Model (B) Won : 12 times
   Static Model (A) Won  : 8 times
   Ties                  : 0 times
============================================================
```
Prompt B performs better as expected, but the experiment must be done with larger data, for meaningful results.


**3. Information Extraction** <br>
From a customer service data set, I compared various prompts to extract data.
I simplified the data set by only using elements that contained the word "order" in its ground truth. <br>
Next steps of this experiment should be done in a clean and larger data set, but since the goal of this repository is to learn different methods, it was enough for me.
Some problems I encountered were: 
- The model not deciding on what information is extraction worthy
- The model extracting the correct information, but in inconsistent format.
#### Approach & Results: <br>
My prompts were promptA (zero-shot), promptB (one-shot) and promptC (few-shots).
    One-shot significantly improved the score, in comparison to one shot, as it taught the model the output format. However there was no massive jump between one-shot and few-shot. The extra computation load might be unnecessary.<br> <br>
    Saved in the [results.md](information-extraction/results.md) file, in the information extraction folder.

**4. Hallucination Reduction** <br>
I ran this test in 2 steps: <br> 
   1. **Small Dataset, Manual Check:** <br>I asked the model to make 10 predictions about finance and business. 5 of the questions   were about real events and products. 
    The other 5 were asking about totally fictional, made up events.
    I compared 2 prompts (A and B) against this challenge. A only asks the model to make predictions about the user's prompt.
    B asks the model to fact check it before answering. <br>
    This version's code is [main_v1.py](hallucination-reduction/main_v1.py). Answers are added into answers_v1.txt file when the code is run. <br> <br>
    2. **Comparing Wikipedia Information with Cosine Similarity:** <br>
    I built a pipeline with Qwen, to gather information from *wikipedia*. Then asked the model to create 1 real 1 fake question from each article. By reusing this pipeline, unique data sets of arbitrary sizes can be created many times. (Default is 50, as used in this project) With the new and large wikipedia [dataset](hallucination-reduction/wikipedia_hallucination_dataset.csv), model is challenged with different prompts. <br> Then, outputs of both prompts are compared with the ground truth by semantic checks (cosine similarity), and added to the CSV file. If the similarity rate is higher than 75%, the answer is considered correct. <br>
    Code of this version is present in [main_v2.py](hallucination-reduction/main_v2.py).
#### Approach & Results: <br> 
By running [hallucination_records.py](hallucination-reduction/hallucination_records.py)
hallucination rates can be seen as below:
    
    Total Trap Questions: 50
    Prompt A Trapped Ratio: 50 / 50 (%100.0 Hallucination Rate)
    Prompt B Trapped Ratio: 27 / 50 (%54.0 Hallucination Rate)
For simplicity, semantic comparison of answers to true questions are excluded. However they are documented in the CSV and available for further study. <br>
Note for future study: By manual inspection, I observed that the correct answer rate of prompt B is higher (20% hallucination), but as it adds extra explanation to "no result find" answers, semantic similarity drops down. For further study, a new prompt should be written for higher performance.

### Usage:
Requirements cover all the modules needed for each part. They should be installed by 
```
pip install requirements.txt
```
in the virtual environment.
Ollama qwen2.5:14b model should be available locally.

### Data Sources & Acknowledgements / Credits

**Dynamic Few-Shot Prompting:** <br> 
The e-commerce customer support data used for the Dynamic Few-Shot experiment is sourced from the Hugging Face dataset [`qgyd2021/e_commerce_customer_service`](https://huggingface.co/datasets/qgyd2021/e_commerce_customer_service) provided by the Hugging Face community.

**Information Extraction:** <br>
This project utilizes the **[jonathansuru/customer_service_information_extraction]** dataset, which is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

* **Original Creator/Author:** jonathansuru
* **Source:** [Customer_Service_Information_Extraction](https://huggingface.co/datasets/jonathansuru/customer_service_information_extraction)

**Hallucination Reduction:** The synthetic dataset used in this project was dynamically generated using factual context retrieved from the **[Wikipedia API](https://en.wikipedia.org/w/api.php)**. 

All original textual content fetched from Wikipedia is available under the [Creative Commons Attribution-ShareAlike License](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License). I would like to thank the Wikimedia Foundation and its contributors for providing open access to human knowledge, which makes AI safety research possible.

=============