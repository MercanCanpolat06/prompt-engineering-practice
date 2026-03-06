import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("dynamic_icl_experiment_results.csv").fillna("")

model = SentenceTransformer('all-MiniLM-L6-v2')

gt_embeddings = model.encode(df['ground_truth'].tolist(), convert_to_tensor=True)
a_embeddings = model.encode(df['answer_A_fewshots'].tolist(), convert_to_tensor=True)
b_embeddings = model.encode(df['answer_B_dynamic'].tolist(), convert_to_tensor=True)

scores_A = []
scores_B = []

for i in range(len(df)):
    score_A = util.cos_sim(gt_embeddings[i], a_embeddings[i]).item()
    score_B = util.cos_sim(gt_embeddings[i], b_embeddings[i]).item()
    
    scores_A.append(score_A)
    scores_B.append(score_B)

df['semantic_score_A'] = scores_A
df['semantic_score_B'] = scores_B

mean_A = np.mean(scores_A)
mean_B = np.mean(scores_B)

b_wins = sum(1 for a, b in zip(scores_A, scores_B) if b > a)
a_wins = sum(1 for a, b in zip(scores_A, scores_B) if a > b)
ties = len(df) - b_wins - a_wins

improvement_pct = ((mean_B - mean_A) / mean_A) * 100

print("\n" + "=" * 60)
print("DYNAMIC IN-CONTEXT LEARNING (SEMANTIC) RESULTS")
print("=" * 60)
print(f"Scenario A (Static One-Shot) Average Score   : {mean_A*100:.2f}%")
print(f"Scenario B (Dynamic Few-Shot) Average Score  : {mean_B*100:.2f}%")
print("-" * 60)
print(f"Performance Improvement (Relative)           : + {improvement_pct:.2f}% better")
print("-" * 60)
print(f"Competition Statistics (Total {len(df)} Questions):")
print(f"   Dynamic Model (B) Won : {b_wins} times")
print(f"   Static Model (A) Won  : {a_wins} times")
print(f"   Ties                  : {ties} times")
print("=" * 60)

