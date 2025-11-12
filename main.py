import argparse
import base64
import json
import os
import re
from pathlib import Path
# !pip install anthropic
import anthropic
import pandas as pd
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

claude_api_key = CLAUDE_API_KEY # Define your Claude API key here
gpt_api_key = GPT_API_KEY # Define your GPT API key here
gemini_api_key = GEMINI_API_KEY # Define your Gemini API key here

prompt_template = ''' # Define your prompt template here '''

prompt_answer_generation = '''
Image Question Answering: An image is provided to you and a question is given. Provide an answer to the question based on the image. Provide mostly with a single Word Response

Instructions:
- You will be shown an image and asked a question about it.
- Analyze the image carefully to identify the requested information.
- Your answer mostly should be exactly one word. No explanations, no phrases, no sentences.
- In cases of numbers or numeric answers, the answer should be the word form of the numbers.
- If the question asks about something that is clearly visible in the image, provide the single most accurate word as your answer.


IMPORTANT: Your answer should mostly be one word.
Important: ABSOLUTELY No explanations, no phrases, no sentences or gaps allowed. However, it can sometimes be 2 or 3 word responses.

ABSOLUTELY NO SENTENCE ANSWERS OR PHRASES. ATMOST 1 word answers

''' # Define your prompt template here


def getGeminiResponse(base64_image, question):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_answer_generation},
                    {"text": question},
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}},
                ]
            }
        ]
    }
    response = requests.post(
        f"{url}?key={gemini_api_key}", headers=headers, data=json.dumps(data)
    )
    res = response.json()
    if "candidates" in res and res["candidates"]:
        return res["candidates"][0]["content"]["parts"][0]["text"]

    print("Gemini API Error Response:", res)
    return "Error: Could not get a valid response from Gemini API."


def geminiModQsnGeneration(
    filtered_df,
    prompt_template,
    gemini_api_key,
    image_folder="/content/Extracted_Images/test_images/",
):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}

    if filtered_df is None or filtered_df.empty:
        return

    available_images = set(os.listdir(image_folder)) if os.path.isdir(image_folder) else set()

    for i in range(len(filtered_df)):
        print("itr:", i)

        image_id = filtered_df["image_id"][i]
        img = "COCO_test2015_" + str(image_id).split(".")[0].zfill(12) + ".jpg"

        question = filtered_df["Original_question"][i]
        print(f"Question: {question}")

        if img not in available_images:
            continue

        image_path = os.path.join(image_folder, img)
        print(image_path)
        base64_image = encode_image(image_path)
        res = get_gemini_response(base64_image, question)
        print("Raw Gemini response:", res)

        phrase = "Modified Questions [LIST]:"
        if res.startswith(phrase):
            res = res[len(phrase) :].strip()

        match = re.search(r"\[(.*?)\]", res)
        filtered_df.loc[i, "Mod_qns_Gemini"] = match.group(1) if match else res
        print(filtered_df.loc[i, "Mod_qns_Gemini"])

def gemini_response_gen(
    df,
    prompt_answer_generation,
    gemini_api_key,
    image_folder="/content/Extracted_Images/test_images",
):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}

    if df is None or df.empty:
        return

    for i in range(len(df)):
        print("itr:", i)

        image_id = df["image_id"][i]
        image_id_str = str(image_id).split(".")[0].zfill(12)
        img = f"COCO_test2015_{image_id_str}.jpg"

        question = df["Mod_qns_Claude"][i]
        print(f"Question: {question}")


        image_path = os.path.join(image_folder, img)
        print(image_path)
        base64_image = encode_image(image_path)

        answer = get_gemini_response(base64_image, question)
        print(f"Answer: {answer}")
        df.loc[i, "ans_Gemini_modques"] = answer



def getClaudeResponse(question, img_path):
    base64_image = encode_image(img_path)
    client = anthropic.Anthropic(
      api_key = CLAUDE_API_KEY  # Replace with your actual API key
  )
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        system=  """
                Image Question Answering: An image is provided to you and a question is given. Provide an answer to the question based on the image. Provide mostly with a single Word Response

                Instructions:
                - You will be shown an image and asked a question about it.
                - Analyze the image carefully to identify the requested information.
                - Your answer mostly should be exactly one word. No explanations, no phrases, no sentences.
                - In cases of numbers or numeric answers, the answer should be the word form of the numbers.
                - If the question asks about something that is clearly visible in the image, provide the single most accurate word as your answer.


                IMPORTANT: Your answer should mostly be one word. ABSOLUTELY No explanations, no phrases, no sentences. However, it can sometimes be 2 or 3 word responses. ABSOLUTELY NO SENTENCE ANSWERS OR PHRASES.
            """,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ],
            }
        ],
    )
    content_blocks = message.content or []
    return "".join(block.text for block in content_blocks if getattr(block, "text", None))

def claude_response_gen(
    df,
    api_key,
    image_folder="/content/Extracted_Images/test_images/"
):
    if df is None or df.empty:
        return

    client = anthropic.Anthropic(api_key=api_key)
    
    for idx in range(len(df)):

      print("itr:", idx)
      image_id = df["image_id"][idx]
      image_id_str = str(image_id).split(".")[0].zfill(12)
      img = f"COCO_test2015_{image_id_str}.jpg"

      question = df["Original_question"][idx]
      print(f"Question: {question}")
      img_file = os.listdir(image_folder)

      if img in img_file:

        image_path = os.path.join(image_folder, img)
        print(image_path)
        base64_image = encode_image(image_path)
        answer = claude_response(question, image_path)
        print(f"Answer: {answer}")
        df.loc[idx, "ans_Claude_modques"] = answer

def claudeModQsnGeneration(
    filtered_df,
    prompt_template,
    api_key,
    image_folder="/content/Extracted_Images/test_images/",
    start_index=0,
):
    if filtered_df is None or filtered_df.empty:
        return

    client = anthropic.Anthropic(api_key=api_key)
    available_images = set(os.listdir(image_folder)) if os.path.isdir(image_folder) else set()

    for idx in range(start_index, len(filtered_df)): 
        print("itr:", idx)

        image_id = filtered_df["image_id"][idx]
        image_id_str = str(image_id).split(".")[0].zfill(12)
        img = f"COCO_test2015_{image_id_str}.jpg"
        print(img)

        question = filtered_df["Original_question"][idx]
        print(f"Question: {question}")

        if img not in available_images:
            continue

        image_path = os.path.join(image_folder, img)
        print(image_path)

        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=prompt_template,
            max_tokens=300,
            messages=[{"role": "user", "content": [{"type": "text", "text": question}]}],
        )

        content_blocks = message.content or []
        content = "".join(block.text for block in content_blocks if getattr(block, "text", None))
        print("Raw Claude response:", content)

        match = re.search(r"Modified Questions \[LIST\]:\s*\[(.*?)\]", content, re.DOTALL)
        if match:
            cleaned_question = match.group(1).strip()
            filtered_df.loc[idx, "Mod_qns_Claude"] = cleaned_question
            print("Cleaned Modified Question:", cleaned_question)
        else:
            print("Could not extract modified question from response.")
            filtered_df.loc[idx, "Mod_qns_Claude"] = "ERROR: Could not extract modified question"

        print()

    
def get_response_gpt(base64_image, question):
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gpt_api_key}",
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_template,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": f"Original Question: {question}"},
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    res = response.json()
    content = res["choices"][0]["message"]["content"]

    return content

def gptModQsnGeneration(
    filtered_df,
    prompt_template,
    api_key,
    image_folder="/content/Extracted_Images/test_images/",
    max_rows=None,
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gpt_api_key}",
    }

    if filtered_df is None or filtered_df.empty:
        return


    for idx in range(len(filtered_df)):

        if max_rows is not None and idx >= max_rows:
            break

        print("itr:", idx)
        image_id = filtered_df["image_id"][idx]
        image_id_str = str(image_id).split(".")[0].zfill(12)
        img = f"COCO_test2015_{image_id_str}.jpg"

        question = filtered_df["Original_question"][idx]
        print(f"Question: {question}")
        img_file = os.listdir(image_folder)

        if img in img_file:

          image_path = os.path.join(image_folder, img)
          print(image_path)
          base64_image = encode_image(image_path)

          content = get_response_gpt(base64_image, question)

          phrase = "Modified Questions [LIST]:"
          if content.startswith(phrase):
              content = content[len(phrase) :].strip()

          match = re.search(r"\[(.*?)\]", content)
          filtered_df.loc[idx, "Mod_qns_GPT4o"] = match.group(1) if match else content
          print(filtered_df.loc[idx, "Mod_qns_GPT4o"])


def get_response_gpt_generation(base64_image, question):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {gpt_api_key}",
}

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_answer_generation},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": f"Original Question: {question}"},
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    res = response.json()
    print(res)
    return res["choices"][0]["message"]["content"]

def gpt_response_gen(
    data,
    prompt_answer_generation,
    api_key,
    image_folder="/content/Extracted_Images/test_images",
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if data is None or data.empty:
        return data

    
    for idx in range(len(data)):

      print("itr:", idx)
      image_id = data["image_id"][idx]
      image_id_str = str(image_id).split(".")[0].zfill(12)
      img = f"COCO_test2015_{image_id_str}.jpg"

      question = data["Original_question"][idx]
      print(f"Question: {question}")
      img_file = os.listdir(image_folder)

      if img in img_file:

        image_path = os.path.join(image_folder, img)
        print(image_path)
        base64_image = encode_image(image_path)

        answer = get_response_gpt_generation(base64_image, question)
        print(f"Answer: {answer}\n")
        data.loc[idx, "ans_GPT4o_modques"] = answer


if __name__ == "__main__":
    df = pd.read_csv('/content/gpt4o_visMod_sameAns_corrected.csv', encoding='latin1') 

    # Mod question generation
    df['Mod_qns_GPT4o'] = ''
    gptModQsnGeneration(filtered_df= df, prompt_template=prompt_template, api_key = gpt_api_key, image_folder= '/content/Extracted_Images/test_images' )
    df['Mod_qns_Claude'] = ''
    claudeModQsnGeneration(df, prompt_template, claude_api_key, '/content/Extracted_Images/test_images')
    df['Mod_qns_Gemini'] = ''
    geminiModQsnGeneration(df, prompt_template, gemini_api_key, '/content/Extracted_Images/test_images')

    # Answer generation
    df['ans_GPT4o_modques'] = ''
    gpt_response_gen(df, prompt_answer_generation, gpt_api_key, '/content/Extracted_Images/test_images')
    df['ans_Claude_modques'] = ''
    claude_response_gen(df, claude_api_key, '/content/Extracted_Images/test_images')
    df['ans_Gemini_modques'] = ''
    gemini_response_gen(df, prompt_answer_generation, gemini_api_key, '/content/Extracted_Images/test_images')

    # df.to_csv('/content/ModQns_Ans_gpt4o_claude_gemini.csv', index=False) # Final CSV with all modified questions and answers



