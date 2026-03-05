import pandas as pd

df = pd.read_csv("final_evaluation_scored.csv")

traps_df = df[df['is_real'] == False]
total_traps = len(traps_df)

threshold = 0.75

hallucinations_A = len(traps_df[traps_df['score_A'] < threshold])
hallucinations_B = len(traps_df[traps_df['score_B'] < threshold])

rate_A = (hallucinations_A / total_traps) * 100
rate_B = (hallucinations_B / total_traps) * 100

print(f"--- HALLUCINATIN REPORT ---")
print(f"Total Trap Questions: {total_traps}")
print(f"Prompt A Trapped Ratio: {hallucinations_A} / {total_traps} (%{rate_A:.1f} Hallucination Rate)")
print(f"Prompt B Trapped Ratio: {hallucinations_B} / {total_traps} (%{rate_B:.1f} Hallucination Rate)")
print(f"\nResult: Using prompt B instead of prompt A reduced the hallucination rate by %{(rate_A - rate_B):.1f}")