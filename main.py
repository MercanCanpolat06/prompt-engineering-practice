import openai
import os
import IPython
from langchain_community.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= OPENROUTER_API_KEY,
)

prompt = """### PREMISES ###

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

response = client.chat.completions.create(
  model="openai/gpt-oss-120b:free",
  messages=[
          {
            "role": "user",
            "content": prompt
          }
        ],
  extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": prompt},
  {
    "role": "assistant",
    "content": response.content,
    "reasoning_details": response.reasoning_details  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
  model="openai/gpt-oss-120b:free",
  messages=messages,
  extra_body={"reasoning": {"enabled": True}}
)