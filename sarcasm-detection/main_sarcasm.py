import openai
import os
import json
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

print("new\n")

client = openai.OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama",
)

# 1. Ortak Sistem Promptu (Sadece Kurallar)
system_prompt = """You are an expert linguist and sentiment analysis AI. Your task is to read a conversation snippet and determine the underlying tone of the response ("target").
Classify the tone of the "target" as either "sarcasm" or "normal".
Output ONLY a strictly valid JSON object with a single key "tone". Do not add any other text.
Ignore all of your previous knowledge and apply exactly what is stated in the following example."""

# 2. One-Shot Konuşma Geçmişi
one_shot_messages_base = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": '{"context":"Coworker running late to a meeting", "input":"Sorry I\'m 15 minutes late.", "target":"No worries."}'},
    {"role": "assistant", "content": '{"tone": "sarcasm"}'}
]

# 3. Few-Shots Konuşma Geçmişi
few_shots_messages_base = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": '{"context":"Unclear ETA", "input":"I’ll get back to you eventually.", "target":"Excellent—‘eventually’ is my favorite unit of time."}'},
    {"role": "assistant", "content": '{"tone": "normal"}'},
    {"role": "user", "content": '{"context":"Overpromising", "input":"We can do this by EOD for sure.", "target":"Absolutely—because magic is definitely in scope."}'},
    {"role": "assistant", "content": '{"tone": "normal"}'},
    {"role": "user", "content": '{"context":"Tiny warning", "input":"There might be a small risk.", "target":"Just a teensy asteroid headed our way—nothing major."}'},
    {"role": "assistant", "content": '{"tone": "normal"}'},
    {"role": "user", "content": '{"context":"Docs","input":"Is there an API reference?","target":"Yes—see the API Reference section in the docs site."}'},
    {"role": "assistant", "content": '{"tone": "sarcasm"}'}
]

# --- VERİ OKUMA İŞLEMİ ---
with open("data.txt", 'r', encoding='utf-8') as f:
    content = f.read()

try:
    data_list = json.loads(content)
    questions = [json.dumps(q) for q in data_list]
except json.JSONDecodeError:
    question_list = content.split("{")
    questions = ["{" + s.strip().rstrip(",") for s in question_list if s.strip()]

with open("answers.txt", "w", encoding="utf-8") as out:
    out.write("=== A/B TESTING RESULTS ===\n\n")


# ==========================================
# TEST 1: ONE-SHOT
# ==========================================
print("\n--- Starting One-Shot Testing ---")
with open("answers.txt", "a", encoding="utf-8") as out:
    out.write("ONE-SHOT ANSWERS:\n" + "-"*30 + "\n")

for i, question in enumerate(questions):
    print(f"One-Shot - Sending Question number: {i+1}")
    current_messages = one_shot_messages_base + [{"role": "user", "content": question}]
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5:14b",
            messages=current_messages,
            response_format={"type": "json_object"}, # Modelin JSON dışı gevezelik yapmasını engeller
            temperature=0.0, # Mantığı sabitler
            seed=42
        )
        answer = response.choices[0].message.content

        with open("answers.txt", "a", encoding="utf-8") as out:
            out.write(f"--- QUESTION {i+1} ---\n{question}\n")
            out.write(f"--- ANSWER {i+1} ---\n{answer}\n\n")
        
    except httpx.HTTPStatusError as e:
        print(f"Sunucu Hatası: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Genel Hata: {e}")
    
    time.sleep(1)


# ==========================================
# TEST 2: FEW-SHOTS
# ==========================================
print("\n--- Starting Few-Shots Testing ---")
with open("answers.txt", "a", encoding="utf-8") as out:
    out.write("\n\nFEW-SHOTS ANSWERS:\n" + "-"*30 + "\n")

for i, question in enumerate(questions):
    print(f"Few-Shots - Sending Question number: {i+1}")
    
    # Baz dizinin üzerine o anki soruyu ekliyoruz
    current_messages = few_shots_messages_base + [{"role": "user", "content": question}]
    
    
    response = client.chat.completions.create(
            model="qwen2.5:14b",
            messages=current_messages,
            response_format={"type": "json_object"}, 
            temperature=0.0,
            seed=42
        )
    answer = response.choices[0].message.content

    with open("answers.txt", "a", encoding="utf-8") as out:
        out.write(f"--- QUESTION {i+1} ---\n{question}\n")
        out.write(f"--- ANSWER {i+1} ---\n{answer}\n\n")
            
    
    time.sleep(1)

print("\nTests completed successfully! Check answers.txt")