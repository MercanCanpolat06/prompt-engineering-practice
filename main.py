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
    base_url="https://router.huggingface.co/v1",
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

one_shot_prompt = """You are an expert logic puzzle solver. You think step-by-step to deduce the correct schedule or arrangement, considering all constraints. 
Always output ONLY a strictly valid JSON object with one key: "answer" (the correct option letter). 

EXAMPLE PUZZLE:
### PREMISES ###
Ali and Berrak stayed at a hotel for five days. For breakfast, the hotel offered crepes, menemen, omelet, pastry, and bagel options. Some information regarding the individuals' food choices is as follows:
- Both individuals chose from all food types.
- The individuals could choose only one type of food per day.
- The food Ali chose on the second day was chosen by Berrak on the fourth day.
- Ali chose crepes, omelet, and pastry on consecutive days, respectively.
- Berrak also chose crepes, omelet, and pastry on consecutive days, respectively.
- Ali chose the bagel on a day prior to the omelet.

### QUESTION ###
Based on this information, which of the following foods could Berrak have chosen on the fifth day?
I. pastry
II. bagel
III. menemen

### OPTIONS ###
[A] Only I
[B] Only II
[C] I and II
[D] II and III
[E] I, II, and III
SOLUTION METHOD:

  "Step 1: Let's denote the foods as C (Crepes), M (Menemen), O (Omelet), P (Pastry), and B (Bagel). 
  Both Ali and Berrak have a strict 3-day consecutive sequence of C -> O -> P. 
  Step 2: Let's find Ali's sequence. Ali's B must be before O. If Ali's sequence started on Day 1 (C, O, P on days 1,2,3), O would be Day 2, meaning B must be Day 1. But Day 1 is C, so this is impossible. 
  If Ali's sequence started on Day 2 (C, O, P on days 2,3,4), Ali's Day 2 is C. 
  Rule says Ali(Day 2) = Berrak(Day 4). So Berrak(Day 4) = C. Since Berrak also has consecutive C -> O -> P, Berrak's Day 5 would be O, but P would need a Day 6, which is impossible. 
  Thus, Ali's C -> O -> P sequence MUST be on Days 3, 4, and 5. 
  Step 3: Now we know Ali is [?, ?, C, O, P]. Ali's Day 1 and 2 are B and M. 
  Since Ali(Day 2) = Berrak(Day 4), Berrak's Day 4 must be either B or M. 
  Step 4: Let's find Berrak's sequence. Berrak has C -> O -> P consecutively. Since Berrak's Day 4 is B or M, Berrak's sequence cannot overlap with Day 4.
  Therefore, Berrak's C -> O -> P MUST be on Days 1, 2, and 3. 
  Step 5: Now we establish the two valid scenarios. 
  Scenario 1: Ali's Day 2 is M. Then Berrak's Day 4 is M. This leaves B for Berrak's Day 5. 
  Scenario 2: Ali's Day 2 is B. Then Berrak's Day 4 is B. This leaves M for Berrak's Day 5. 
  Step 6: Conclusion. On the 5th day, Berrak could have chosen B (Bagel) or M (Menemen). Looking at the Roman numerals, Bagel is II and Menemen is III.",

EXAMPLE ANSWER:
{
  "answer": "D"
}"""

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
          model="meta-llama/Llama-3.1-8B-Instruct",
          messages=[
                  {
                    "role": "user",
                    "content":  zero_shot_prompt + question
                  }
                ],
          #extra_body={"reasoning": {"enabled": True}},
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
          model="meta-llama/Llama-3.1-8B-Instruct",
          messages=[
                  {
                    "role": "user",
                    "content":  one_shot_prompt + question
                  }
                ],
          #extra_body={"reasoning": {"enabled": True}},
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