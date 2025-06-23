import os
import time
from typing import cast

import pandas as pd
import requests
from prompt import build_prompt

DATA_FILE = "../data.csv"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")

OPENROUTER_MODELS = {
    "deepseek-r1-distill-llama-70b": "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek-r1": "deepseek/deepseek-r1:free",
    "gemma_3_27b": "google/gemma-3-27b-it:free",
    "llama-4-maverick": "meta-llama/llama-4-maverick:free",
    "gemini-25-flash-preview-05-20": "google/gemini-2.5-flash-preview-05-20",
    "gemini-25-flash-lite-preview-06-17": "google/gemini-2.5-flash-lite-preview-06-17",
    "mistral-7b-instruct": "mistralai/mistral-7b-instruct:free",
    "llama-33-70b-instruct": "meta-llama/llama-3.3-70b-instruct:free",
    "gpt-41-nano": "openai/gpt-4.1-nano",
    "gpt-41-mini": "openai/gpt-4.1-mini",
}

MODELS_TO_TEST = [
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1",
    "gemma_3_27b",
    "llama-4-maverick",
    "gemini-25-flash-preview-05-20",
    "gemini-25-flash-lite-preview-06-17",
    "mistral-7b-instruct",
    "llama-33-70b-instruct",
    "gpt-41-nano",
    "gpt-41-mini",
]


class OpenRouterClient:
    def __init__(self, api_key: str):
        if not api_key or api_key == "your_openrouter_api_key_here":
            raise ValueError("Please set your OpenRouter API key!")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def query_model(self, question: str, context: str, model_id: str) -> str | None:
        prompt = build_prompt(question, context)

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.0,
            "top_p": 0.1,
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()

                elif response.status_code == 429:
                    time.sleep(2**attempt)
                    continue

                else:
                    print(f"API Error {response.status_code}")
                    break

            except Exception as e:
                print(f"Request error: {e}")
                if attempt < 2:
                    time.sleep(1)

        return None


def main():
    if OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        print("Please set your OpenRouter API key")
        return

    df = pd.read_csv(DATA_FILE, sep="\t", encoding="utf-8")

    required_cols = ["Question", "Document"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return

    client = OpenRouterClient(OPENROUTER_API_KEY)

    for model_name in MODELS_TO_TEST:
        print(f"{model_name}: Starting predictions...")

        model_id = OPENROUTER_MODELS[model_name]
        column_name = f"{model_name}_prediction"

        predictions = []
        for idx, row in df.iterrows():
            if pd.isna(row["Question"]) or pd.isna(row["Document"]):
                predictions.append(None)
                continue

            prediction = client.query_model(
                question=str(row["Question"]),
                context=str(row["Document"]),
                model_id=model_id,
            )

            predictions.append(prediction)

            index = cast(int, idx)

            if (index + 1) % 10 == 0:
                print(f"{model_name}: {index + 1} / {len(df)}")

        df[column_name] = predictions

        not_in_document_count = 0
        valid_predictions = 0

        for idx, row in df.iterrows():
            prediction = row[column_name]
            document = str(row["Document"])

            if pd.notna(prediction) and prediction.strip():
                valid_predictions += 1
                if prediction.strip() not in document:
                    not_in_document_count += 1

        df.to_csv(DATA_FILE, index=False, sep="\t", encoding="utf-8")

        print(df.head())

        successful = sum(1 for p in predictions if p is not None)
        print(f"{model_name}: Complete ({successful} / {len(df)} successful)")
        print(
            f"{model_name}: {not_in_document_count} / {valid_predictions} predictions not found in document ({not_in_document_count / valid_predictions * 100:.1f}%)"
            if valid_predictions > 0
            else f"{model_name}: No valid predictions to check"
        )


if __name__ == "__main__":
    main()
