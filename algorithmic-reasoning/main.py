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


one_shot_prompt = """
You are an expert logic puzzle solver algorithm. You must strictly follow a 4-stage deductive reasoning process inside <thinking> tags before providing the final answer.

YOUR 4-STAGE PROCESS:
[STAGE 1: ENTITIES] List all variables and available slots (e.g., Seats 1 to 6).
[STAGE 2: CONSTRAINTS] Separate absolute facts from relative conditions. 
[STAGE 3: STRICT SLOT MAPPING] 
- Step 1: Place the absolute constraints first. 
- Step 2: WRITE DOWN the EXACT numbers of the remaining EMPTY slots. CRITICAL: You are strictly forbidden from placing anyone in a slot that is already occupied.
- Step 3: Apply relative rules. Words like "between" or "next to" mean they must sit in strictly consecutive EMPTY slots. (Note: If it's a round table, seats 6 and 1 are consecutive).
- Step 4: Branch out into explicit scenarios (Scenario A, Scenario B) to test where the remaining people can fit.
[STAGE 4: EVALUATION] Cross-check the options against ALL valid scenarios to find what MUST be true or COULD be true.

After closing the </thinking> tag, you must output ONLY a valid JSON object: {"answer": "Letter"}.
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
          seed=42,                   
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
        time.sleep(1)

print("Chain of Thought Testing")

with open("answers.txt", "a", encoding="utf-8") as out:
     out.write("Chain of Thought Answers:\n")

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
        time.sleep(1)