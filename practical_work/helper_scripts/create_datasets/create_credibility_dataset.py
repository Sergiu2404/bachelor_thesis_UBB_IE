# import requests
# import csv
# import random
# import time
#
# API_KEY = "20e3ba24e51428932aea68059203d5154a1119c7e477de5463a1e513e3c999db"
# API_URL = "https://api.together.xyz/v1/chat/completions"
# MODEL = "meta-llama/Llama-3-8b-chat-hf"
#
# def generate_text(prompt):
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.7,
#         "max_tokens": 300
#     }
#     for _ in range(3):
#         try:
#             response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
#             response.raise_for_status()
#             return response.json()["choices"][0]["message"]["content"].strip()
#         except Exception as e:
#             print(f"Error: {e}")
#             time.sleep(5)
#     return ""
#
# def generate_dataset(prompt, score_range, count):
#     data = []
#     for i in range(count):
#         print(f"Generating {i + 1}/{count}")
#         text = generate_text(prompt)
#         if text:
#             score = round(random.uniform(*score_range), 2)
#             data.append({"text": text, "credibility_score": score})
#         else:
#             print("Failed to generate text.")
#         time.sleep(1)
#     return data
#
# def save_to_csv(data, filename):
#     with open(filename, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=["text", "credibility_score"])
#         writer.writeheader()
#         for row in data:
#             writer.writerow(row)
#     print(f"Saved {len(data)} entries to {filename}")
#
# def create_dataset():
#     credible_data = generate_dataset(credible_prompt, (0.8, 1.0), 1000)
#     fake_data = generate_dataset(fake_prompt, (0.0, 0.2), 1000)
#     all_data = credible_data + fake_data
#     random.shuffle(all_data)
#     save_to_csv(all_data, "credibility_datasets/news_credibility_dataset.csv")
#
# if __name__ == "__main__":
#     create_dataset()




import requests
import csv
import random
import time
import os

API_KEY = "20e3ba24e51428932aea68059203d5154a1119c7e477de5463a1e513e3c999db"
API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL = "meta-llama/Llama-3-8b-chat-hf"

def generate_text(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 300
    }
    for _ in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    return ""

def generate_dataset(prompt, score_range, count):
    data = []
    for i in range(count):
        print(f"Generating {i + 1}/{count}")
        text = generate_text(prompt)
        if text:
            score = round(random.uniform(*score_range), 2)
            data.append({"text": text, "credibility_score": score})
        else:
            print("Failed to generate text.")
        time.sleep(1)
    return data

def save_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "credibility_score"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Saved {len(data)} entries to {filename}")

def create_dataset():
    # credible_prompt = (
    #     "Write a fact-checked financial news article (3-5 sentences) about global markets or corporations or global financial situation."
    #     "Use generic terms like 'the company', 'a corporation', 'global markets', etc."
    #     "The article must be realistic, professionally written, and reflect real financial trends or economic data."
    #     "Include a title and a body."
    # )
    #
    # fake_prompt = (
    #     "Write a fake or misleading financial news article (3-5 sentences) that sounds plausible but contains factual errors."
    #     "Use generic company terms like 'a corporation', 'the business', etc."
    #     "It should reflect typical financial misinformation, such as exaggerated claims, conspiracy theories, or market manipulation rumors."
    #     "Include a title and a body."
    # )
    credible_prompt = (
        "Generate a fact-checked financial news article that meets the following criteria:"
        "\n1. Length: 3-5 sentences"
        "\n2. Content Requirements:"
        "\n   - Base the article on verifiable economic data or recent market reports"
        "\n   - Use specific, but anonymized corporate references (e.g., 'a leading tech company', 'a multinational financial institution')"
        "\n   - Include at least one quantitative economic indicator or market statistic"
        "\n3. Tone and Style:"
        "\n   - Maintain a neutral, professional journalistic tone"
        "\n   - Avoid sensationalism or speculative language"
        "\n   - Reflect current global economic trends or market conditions"
        "\n4. Structure:"
        "\n   - Provide a clear, concise title that summarizes the key information"
        "\n   - Ensure the body presents a balanced, factual narrative"
        "\n5. Credibility Markers:"
        "\n   - Demonstrate logical reasoning"
        "\n   - Use precise language"
        "\n   - Indicate potential sources or types of data used (without citing specific sources)"
    )

    fake_prompt = (
        "Create a misleading financial news article that demonstrates characteristics of misinformation:"
        "\n1. Length: 3-5 sentences"
        "\n2. Misinformation Techniques:"
        "\n   - Incorporate at least one of these deceptive strategies:"
        "\n     a) Exaggerated market predictions"
        "\n     b) Conspiracy theory elements"
        "\n     c) Misleading statistical interpretation"
        "\n     d) Fabricated market manipulation claims"
        "\n3. Content Characteristics:"
        "\n   - Use vague, emotionally charged language"
        "\n   - Create plausible-sounding but fictitious market scenarios"
        "\n   - Hint at insider knowledge without substantive evidence"
        "\n4. Structural Elements:"
        "\n   - Craft a sensationalist title that implies significant market impact"
        "\n   - Use generic corporate references"
        "\n   - Include pseudo-authoritative language"
        "\n5. Subtlety in Deception:"
        "\n   - Ensure the article sounds somewhat believable"
        "\n   - Mix small truths with significant fabrications"
        "\n   - Avoid overtly ridiculous or immediately detectible falsehoods"
    )

    credible_data = generate_dataset(credible_prompt, (0.8, 1.0), 1000)
    fake_data = generate_dataset(fake_prompt, (0.0, 0.2), 1000)
    all_data = credible_data + fake_data
    random.shuffle(all_data)
    save_to_csv(all_data, "credibility_datasets/financial_news_credibility_dataset.csv")

if __name__ == "__main__":
    create_dataset()
