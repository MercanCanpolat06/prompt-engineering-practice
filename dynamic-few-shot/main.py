import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

df = pd.read_json("faq.json")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

pool_df = df.iloc[:45].reset_index(drop=True)
test_df = df.iloc[45:].reset_index(drop=True)

pool_df.to_csv("ecommerce_pool_set.csv", index=False)
test_df.to_csv("ecommerce_test_set.csv", index=False)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

pool_questions = pool_df['question'].tolist()
pool_embeddings = embed_model.encode(pool_questions)

dimension = pool_embeddings.shape[1] 
index = faiss.IndexFlatL2(dimension)
index.add(pool_embeddings)

faiss.write_index(index, "ecommerce_faq.index")
print("Questions are added to faiss")

def query_ollama(prompt, system_prompt):
    payload = {
        "model": "qwen2.5:14b",
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4, 
            "seed": 42
        }
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    if response.status_code == 200:
        return response.json()['response'].strip()
    else:
        return f"Error: {response.status_code}"
    
results = []

static_context = ""
for i in range(3):
    fixed_q = pool_df.iloc[i]['question']
    fixed_a = pool_df.iloc[i]['answer']
    static_context += f"Question {i+1} : {fixed_q}\nAnswer {i+1} : {fixed_a}\n\n"

for idx, row in test_df.iterrows():
    test_question = row['question']
    ground_truth = row['answer']
    
    print(f"[{idx+1}] Test Question: {test_question}")
    
    test_vector = embed_model.encode([test_question])

    D, I = index.search(test_vector, k=3) 

    dynamic_context = ""
    for i, pool_idx in enumerate(I[0]):
        similar_q = pool_df.iloc[pool_idx]['question']
        similar_a = pool_df.iloc[pool_idx]['answer']
        dynamic_context += f"Example {i+1}:\Question: {similar_q}\Answer: {similar_a}\n\n"

    # few static shot prompt
    system_A = (
        "You are an expert customer service agent for an e-commerce website. "
        "Use the provided past examples to understand the company policies and the expected tone. "
        "Answer the new User Question based strictly on the style and rules shown in the examples.\n\n"
        "Past Examples:\n"
        f"{static_context}"
    )
    user_prompt_A = f"User Question: {test_question}\nAnswer:"
    
    answer_A = query_ollama(user_prompt_A, system_A)
    
    # dynamic prompt
    system_B = (
        "You are an expert customer service agent for an e-commerce website. "
        "Use the provided past examples to understand the company policies and the expected tone. "
        "Answer the new User Question based strictly on the style and rules shown in the examples."
    )
    user_prompt_B = f"--- PAST EXAMPLES ---\n{dynamic_context}\n\n--- NEW TASK ---\nUser Question: {test_question}\nAnswer:"
    
    answer_B = query_ollama(user_prompt_B, system_B)
    
    results.append({
        "test_question": test_question,
        "ground_truth": ground_truth,
        "retrieved_context": dynamic_context,
        "answer_A_fewshots": answer_A,
        "answer_B_dynamic": answer_B
    })

results_df = pd.DataFrame(results)
results_df.to_csv("dynamic_icl_experiment_results.csv", index=False)

print("\n'dynamic_icl_experiment_results.csv' is created.")