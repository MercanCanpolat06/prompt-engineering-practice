import openai
import os
import IPython
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import time
import httpx

load_dotenv()

OPENROUTER_API_KEY = os.getenv("HugginFace_API_KEY")

client = openai.OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama",
)

prompt = """Solve the following logic puzzle and output the result ONLY in a strictly valid JSON format. No conversational text, no explanations.

### PREMISES ###

There are three people named Ahmet, Gözde, and Jale. 
They bought six types of fish: red mullet, bluefish, anchovy, sea bass, coral, and bonito. 
They cooked these fish either on the grill or in a pan. 
Everyone bought exactly two different types of fish. 
Everyone cooked one of their fish in a pan and the other on the grill. 
Red mullet and bluefish were bought by different people and were cooked using different methods. 
Each type of fish was bought exactly once. 
Gözde bought anchovy and cooked it on the grill. 
Ahmet bought sea bass and bonito.

### QUESTION ###
Based on the information above, which of the following fish could have been cooked on the grill? 
I. Bluefish 
II. Sea bass
III. Coral

### OPTIONS ###
[A] Only I
[B] Only II
[C] I and II
[D] II and III
[E] I, II, and III"""

one_shot_prompt = """You are an expert logic puzzle solver algorithm. You must strictly follow a 4-stage deductive reasoning process inside <thinking> tags before providing the final answer.

YOUR 4-STAGE PROCESS:
[STAGE 1: ENTITIES] List all categories, items, and their strict limits.
[STAGE 2: CONSTRAINTS] Separate absolute facts (exact placements) from relative conditions.
[STAGE 3: MAPPING] Build the scenarios step-by-step. Start with absolute facts. Apply relative constraints to find all possible valid layouts. 
[STAGE 4: EVALUATION] Test the given options against your valid layouts to answer the specific question asked.

After completing the <thinking> block, output ONLY a valid JSON object: {"answer": "Letter"}

### EXAMPLE PUZZLE ###
Tasks X, Y, and Z are performed on Monday, Tuesday, and Wednesday. 
- Task Y is performed on Wednesday.
- Task X is performed on a day after Task Z.
QUESTION: Which task is performed on Tuesday?
OPTIONS:
[A] X
[B] Y
[C] Z

### EXAMPLE THINKING PROCESS ###
[STAGE 1: ENTITIES]
- Days: Monday, Tuesday, Wednesday (Chronological).
- Tasks: X, Y, Z (Each performed exactly once).

[STAGE 2: CONSTRAINTS]
- Absolute: Y = Wednesday.
- Relative: Z < X (Z happens before X).

[STAGE 3: MAPPING]
- Place Absolute: _ , _ , Y. (Monday and Tuesday are empty).
- Apply Relative: The remaining tasks are X and Z. The remaining days are Monday and Tuesday. 
- Since Z must happen before X, Z must be on Monday and X must be on Tuesday.
- Final Layout: Monday=Z, Tuesday=X, Wednesday=Y.

[STAGE 4: EVALUATION]
- The question asks which task is on Tuesday.
- Based on the final layout, Task X is on Tuesday. This matches option [A].

### EXAMPLE ANSWER ###
{
  "answer": "A"
}
"""

zero_shot_prompt = "Solve the following logic puzzle and output the result ONLY in a strictly valid JSON format. No conversational text, no explanations."

# Chat loop
with open("questions.txt", 'r', encoding='utf-8') as f:
        content = f.read()

question_list = content.split("---")
questions = [s.strip() for s in question_list if s.strip()]

print(" Zero - Shot Testing:")

with open("answers.txt", "a", encoding="utf-8") as out:
     out.write("Zero Shot Answers:\n")

for i, question in enumerate(questions):
        print(f"Sending Question number: {i}")
        try:
          response = client.chat.completions.create(
          model="qwen2.5:14b",
          messages=[
                  {
                    "role": "user",
                    "content":  zero_shot_prompt + question
                  }
                ],
          #extra_body={"reasoning": {"enabled": True}},
          temperature=0.0,      
          top_p=1.0,            
          seed=42,              # Deneylerin tekrarlanabilirliği için sabit bir tohum
          max_tokens=2000,     
          frequency_penalty=0.
          )
          response = response.choices[0].message.content

          with open("answers.txt", "a", encoding="utf-8") as out:
                out.write(f"--- QUESTION {i+1} ---\n")
                out.write(f"--- ANSWER {i+1} ---\n{response}\n")
                out.write("\n" + "="*50 + "\n\n")
            
          print(f"Question {i} completed.")
        except httpx.HTTPStatusError as e:
            print(f"Sunucu Hatası: {e.response.status_code}")
            print(f"Sunucunun Gerçek Cevabı: {e.response.text}") # JSONDecodeError'a sebep olan o metni burada göreceğiz
        except Exception as e:
            print(f"Genel Hata: {e}")
        time.sleep(3)

print("One-Shot Testing")

with open("answers.txt", "a", encoding="utf-8") as out:
     out.write("One Shot Answers:\n")

for i, question in enumerate(questions):
        print(f"Sending Question number: {i}")
        try:
          response = client.chat.completions.create(
          model="qwen2.5:14b",
          messages=[
                  {
                    "role": "user",
                    "content":  one_shot_prompt + question
                  }
                ],
          #extra_body={"reasoning": {"enabled": True}},
          temperature=0.0,      
          top_p=1.0,            
          seed=42,              
          max_tokens=2000,     
          frequency_penalty=0.
          )
          response = response.choices[0].message.content

          with open("answers.txt", "a", encoding="utf-8") as out:
                out.write(f"--- QUESTION {i+1} ---\n")
                out.write(f"--- ANSWER {i+1} ---\n{response}\n")
                out.write("\n" + "="*50 + "\n\n")
            
          print(f"Question {i} completed.")
        except httpx.HTTPStatusError as e:
            print(f"Sunucu Hatası: {e.response.status_code}")
            print(f"Sunucunun Gerçek Cevabı: {e.response.text}") # JSONDecodeError'a sebep olan o metni burada göreceğiz
        except Exception as e:
            print(f"Genel Hata: {e}")
        time.sleep(3)