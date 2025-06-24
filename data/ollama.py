import os
from typing import cast

import pandas as pd
import requests

from prompt import build_prompt

DATA_FILE = "../data.csv"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

MACEDONIAN_MODELS = {
    "domestic_yak_8b": "hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0",
}

MODELS_TO_TEST = ["domestic_yak_8b"]


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_HOST) -> None:
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def query_model(self, question: str, context: str, model_name: str) -> str | None:
        prompt = build_prompt(question, context)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 200,
                "top_p": 0.1,
                "stop": ["\n\nПрашање:", "\n\nДокумент:", "\n\nВАЖНО:"],
            },
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            print(f"Error querying {model_name}: {e}")

        return None


def main() -> None:
    df = pd.read_csv(DATA_FILE, sep="\t", encoding="utf-8")

    required_cols = ["Question", "Document"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return

    client = OllamaClient()

    for model_name in MODELS_TO_TEST:
        print(f"{model_name}: Starting predictions...")

        ollama_model_id = MACEDONIAN_MODELS[model_name]
        column_name = f"{model_name}_prediction"

        predictions: list[str | None] = []
        for idx, row in df.iterrows():
            if pd.isna(row["Question"]) or pd.isna(row["Document"]):
                predictions.append(None)
                continue

            prediction = client.query_model(
                question=str(row["Question"]),
                context=str(row["Document"]),
                model_name=ollama_model_id,
            )

            predictions.append(prediction)

            index = cast(int, idx)

            if (index + 1) % 10 == 0:
                print(f"{model_name}: {index + 1} / {len(df)}")

        df[column_name] = predictions

        not_in_document_count = 0
        valid_predictions = 0

        for _, row in df.iterrows():
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
            else f"{model_name}: No valid predictions to check",
        )


if __name__ == "__main__":
    main()
