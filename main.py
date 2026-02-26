import openai
import os
import IPython
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= OPENROUTER_API_KEY,
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


# Chat loop
with open("questions.txt", 'r', encoding='utf-8') as f:
        content = f.read()

question_list = content.split("---")
questions = [s.strip() for s in question_list if s.strip()]

for i, question in enumerate(questions):
        print(f"Sending Question number: {i}")
        try:
          response = client.chat.completions.create(
          model="openai/gpt-oss-120b:free",
          messages=[
                  {
                    "role": "user",
                    "content": "Solve the following logic puzzle and output the result ONLY in a strictly valid JSON format. No conversational text, no explanations." + question
                  }
                ],
          extra_body={"reasoning": {"enabled": True}}
          )
          response = response.choices[0].message

          with open("answers.txt", "a", encoding="utf-8") as out:
                out.write(f"--- QUESTION {i+1} ---\n")
                out.write(f"--- ANSWER {i+1} ---\n{response}\n")
                out.write("\n" + "="*50 + "\n\n")
            
          print(f"Question {i+1} completed.")
        except:
              print(f"Error in sending question {i}")
        time.sleep(3)
