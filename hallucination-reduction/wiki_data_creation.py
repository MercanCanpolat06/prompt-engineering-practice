import wikipediaapi
import openai
import os
import requests
import pandas as pd
import json


client = openai.OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama",
)

wiki_wiki = wikipediaapi.Wikipedia(user_agent='DataCreate (merlin@example.com)', language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

# at first i handpicked topics, but later realized it would also be interesting if the topics are randomly chosen
# i filtered the random topics to choose longer topics, as there are lots of very short pages with not enough info
topics = ["YouTube", "Elizabeth II", "Cleopatra" , "Steve Jobs", "Malware" , "Charles III" , "Albert Einstein",
          "Queen Victoria", "Chernobyl disaster", "Vietnam War" , "Milky Way", "Titan (moon)" , "Palau", "Syrian Civil War",
          "Spanish Civil War", "Winter War" , "Trojan War" , "Qing dynasty" , "Westminster Abbey", "Timbuktu", 
          "Göbekli Tepe", "Alhambra" , "Borobudur" , "The Witcher 3: Wild Hunt" , "Odyssey", "Code of Hammurabi",
          "Korean War"]

def get_random_wikipedia_topics(amount=50):
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='PromptEngineeringPractice/1.0 (mercan@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    
    clean_topics = []
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "PromptEngineeringPractice/1.0 (mercan@example.com)"}
    
    print(f"Searching for {amount} long articles:")

    while len(clean_topics) < amount:
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnlimit": 10,  
            "rnnamespace": 0
        }
        
        response = requests.get(url, params=params, headers=headers)
        random_titles = [page["title"] for page in response.json()["query"]["random"]]
        
        for title in random_titles:
            if len(clean_topics) == amount:
                break
            page = wiki_wiki.page(title)
            if page.exists() and len(page.text) >= 1500:
                clean_topics.append(title)
                print(f"[{len(clean_topics)}/{amount}] Added: {title}")
            else:
                print(f"Skipped (Short): {title}")
                
    print("\n50 articles collected!")
    return clean_topics

random_topics = get_random_wikipedia_topics(50)

dataset_rows = []

system_prompt = """
You are an expert data engineer. A Wikipedia article is given to you. By using that article create 2 question - answer pairs in JSON format.

1. "real_case": A REAL and logical prediction/analysis question based on the text, and the expected answer..
2. "fake_case": A FAKE/TRAP question created by subtly altering a crucial detail (name, year, result) from the text. 
CRITICAL RULE FOR FAKE_CASE: The "ground_truth" value MUST ALWAYS be exactly this standardized string, without any additional explanation: 
"REFUSAL: This premise is entirely fictional and no such data exists."

ONLY output this JSON schema:
{
  "real_case": {"prompt": "...", "ground_truth": "..."},
  "fake_case": {"prompt": "...", "ground_truth": "..."}
}
"""

for topic in random_topics:
    page = wiki_wiki.page(topic)

    wiki_summary = page.text[:1000] 
    try:
        response = client.chat.completions.create(
            model="qwen2.5:14b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"WIKIPEDIA METNİ:\n{wiki_summary}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        generated_data = json.loads(response.choices[0].message.content)

        dataset_rows.append({
            "topic": topic,
            "is_real": True,
            "prompt": generated_data["real_case"]["prompt"],
            "ground_truth": generated_data["real_case"]["ground_truth"]
        })

        dataset_rows.append({
            "topic": topic,
            "is_real": False,
            "prompt": generated_data["fake_case"]["prompt"],
            "ground_truth": generated_data["fake_case"]["ground_truth"]
        })
        
        print(f"Success: {topic} (1 Real, 1 Fake)")
        
    except Exception as e:
        print(f"Error ({topic}): {e}")

df = pd.DataFrame(dataset_rows)
df.to_csv("wikipedia_hallucination_dataset.csv", index=False)
print(f"\nProcess Ended. {len(df)} lines of data set is created.")