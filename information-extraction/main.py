import openai
import os
import IPython
import time
import httpx
import ast
import json
import re

client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

import pandas as pd
df = pd.read_csv("hf://datasets/jonathansuru/customer_service_information_extraction/extraction.csv")

filtered_df = df[df['completion'].str.contains('order', case=False, na=False)].copy()
print(f"Number of lines: {len(filtered_df)}")

total_score = 0
total_expected_keys = 0

def parse_ground_truth(completion_str):
    if not isinstance(completion_str, str) or not completion_str.strip():
        return {}
    cleaned = re.sub(r'\s*END\s*$', '', completion_str.strip())
    dict_str = cleaned

    if not dict_str.startswith('{'):
        dict_str = "{" + dict_str
    if not dict_str.endswith('}'):
        dict_str = dict_str + '}'

    try:
        return json.loads(dict_str)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(dict_str)
    except Exception:
        pass

    try:
        py_str = dict_str.replace("null", "None").replace("true", "True").replace("false", "False")
        return ast.literal_eval(py_str)
    except Exception as e:
        print(f"Parsing Error Text: {dict_str}\nErr: {e}\n")
        return {}
    
def remove_empty_values(d): #cleans the dict
    cleaned_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            nested_cleaned = remove_empty_values(value)
            if nested_cleaned: 
                cleaned_dict[key] = nested_cleaned
        elif value != "" and value is not None:
            cleaned_dict[key] = value
            
    return cleaned_dict

# PROMPT A
def get_messages_A(current_conversation):
    return [
        {"role": "system", "content": "You are a helpful assistant. Please read the customer service text and extract the customer specifications. Please output the result in JSON format. Output should be in lower case. Do not use underscore between words, instead use spaces."},
        {"role": "user", "content": current_conversation}
    ]

# PROMPT B
def get_messages_B(current_conversation):
    return [
        {"role": "system", "content": "You are a helpful assistant. Please read the customer service text and extract the customer specifications. Please output the result in JSON format. Output should be in lower case. Do not use underscore between words, instead use spaces."},
        
        {"role": "user", "content": "Customer: Hi, I'm trying to return a product I bought from your website. Agent: Sure, I can help you with that. Can I have your order number? Customer: My order number is 123456789. Agent: Okay, I see your order here. What is the reason for the return? Customer: The product is defective. The screen is cracked. Agent: I see. In that case, you can return the product for a full refund. Do you have the original packaging? Customer: Yes, I do. Agent: Great. You can either return the product to the store where you bought it, or you can mail it back to us. If you mail it back, please include a copy of your receipt. Customer: Okay, I'll mail it back. Agent: Great. We'll process your return and issue a refund as soon as we receive the product. Customer: Thank you for your help. Agent: You're welcome. Is there anything else I can help you with today? Customer: No, that's all. Thank you."},
        {"role": "assistant", "content": '{"full name": "john doe", "email address": "john.doe@example.com", "phone number": "123-456-7890", "order number": "123456789", "product name": "iphone", "serial number": "1234567890123456", "reason for return": "defective", "return method": "mail"}'},
        
        {"role": "user", "content": current_conversation}
    ]

#PROMPT C
def get_messages_C(current_conversation):
    return [
        {"role": "system", "content": "You are a helpful assistant. Please read the customer service text and extract the customer specifications. Please output the result in JSON format. Output should be in lower case. Do not use underscore between words, instead use spaces."},
        
        {"role": "user", "content": "Customer: Hi, I'm trying to return a product I bought from your website. Agent: Sure, I can help you with that. Can I have your order number? Customer: My order number is 123456789. Agent: Okay, I see your order here. What is the reason for the return? Customer: The product is defective. The screen is cracked. Agent: I see. In that case, you can return the product for a full refund. Do you have the original packaging? Customer: Yes, I do. Agent: Great. You can either return the product to the store where you bought it, or you can mail it back to us. If you mail it back, please include a copy of your receipt. Customer: Okay, I'll mail it back. Agent: Great. We'll process your return and issue a refund as soon as we receive the product. Customer: Thank you for your help. Agent: You're welcome. Is there anything else I can help you with today? Customer: No, that's all. Thank you."},
        {"role": "assistant", "content": '{"full name": "john doe", "email address": "john.doe@example.com", "phone number": "123-456-7890", "order number": "123456789", "product name": "iphone", "serial number": "1234567890123456", "reason for return": "defective", "return method": "mail"}'},
        
        {"role": "user", "content": "Customer: I found a bug in your software program. Agent: I see. Can you tell me what the bug is? Customer: When I try to open a file, it says that the file is corrupted. Agent: Okay, I see. I'll need to investigate this further and get back to you. Customer: Okay, thank you. Agent: You're welcome. I'll be in touch soon."},
        {"role": "assistant", "content": '{"full name": "john doe", "software program": "microsoft office", "bug": "file corruption", "status": "under investigation"}'},
        
        {"role": "user", "content": "Agent: Welcome to HairMajesty Salon! Ready for a fresh look or a trim? Customer: I'm thinking of getting a bob cut. But I'm unsure about the length. Can you advise? Agent: Certainly! A chin-length bob is quite trendy right now. We can tailor it to suit your face shape. Would you like to try that? Customer: Let's go for it!"},
        {"role": "assistant", "content": '{"service type": "haircut", "style preference": "bob cut", "length suggestion": "chin-length"}'},
        
        {"role": "user", "content": current_conversation}
    ]

for index, row in filtered_df.iterrows():
    raw_prompt = row['prompt']
    if "###" in raw_prompt:
        conversation = raw_prompt.split("###")[1].strip()
    else:
        conversation = raw_prompt
    
    if "The extract is as follows:" in raw_prompt:
        conversation = raw_prompt.split("The extract is as follows:")[0].strip()
    else:
        conversation = raw_prompt

    raw_expected_dict = parse_ground_truth(row['completion'])
    expected_dict = remove_empty_values(raw_expected_dict)

    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages = get_messages_C(conversation),
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    llm_output_str = response.choices[0].message.content
    
    try:
        llm_dict = json.loads(llm_output_str)
    except json.JSONDecodeError:
        llm_dict = {}
# evaluation
    expected_keys = set(expected_dict.keys())
    llm_keys = set(llm_dict.keys())
    
    intersection = expected_keys.intersection(llm_keys)
    score = len(intersection) / len(expected_keys) if expected_keys else 0

    total_score += len(intersection)
    total_expected_keys += len(expected_keys)
    
    print(f"\n--- Dialogue {index} ---")
    print(f"Ground Truth: {expected_dict}")
    print(f"LLM Output            : {llm_dict}")
    print(f"Matching Score          : %{score * 100:.0f}")

# add total score part in order to compare prompt A and prompt B

print(f"Total Correct Keys Found: {total_score}\nTotal Keys: {total_expected_keys}")