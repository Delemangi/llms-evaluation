import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from prompt import build_prompt

DATA_FILE = "../data.csv"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")

OPENROUTER_MODELS = {
    "deepseek-r1-distill-llama-70b": "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek-r1": "deepseek/deepseek-r1",
    "gemma_3_27b": "google/gemma-3-27b-it",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "gemini-25-flash-preview-05-20": "google/gemini-2.5-flash-preview-05-20",
    "gemini-25-flash-lite-preview-06-17": "google/gemini-2.5-flash-lite-preview-06-17",
    "mistral-7b-instruct": "mistralai/mistral-7b-instruct",
    "llama-33-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "gpt-41-nano": "openai/gpt-4.1-nano",
    "gpt-41-mini": "openai/gpt-4.1-mini",
    "o4-mini": "openai/o4-mini",
    "gemini-20-flash-001": "google/gemini-2.0-flash-001",
    "gemini-20-flash-lite-001": "google/gemini-2.0-flash-lite-001",
    "mistral-nemo": "mistralai/mistral-nemo",
    "qwen-25-72b-instruct": "qwen/qwen-2.5-72b-instruct",
}

REASONING_MODELS = ["o4-mini", "deepseek-r1", "deepseek-r1-distill-llama-70b"]

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
    "o4-mini",
    "gemini-20-flash-001",
    "gemini-20-flash-lite-001",
    "mistral-nemo",
    "qwen-25-72b-instruct",
]

MAX_WORKERS = 5
BATCH_SIZE = 20


class OpenRouterClient:
    def __init__(self, api_key: str) -> None:
        if not api_key or api_key == "your_openrouter_api_key_here":
            raise ValueError("Please set your OpenRouter API key!")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.lock = threading.Lock()

    def query_model(
        self,
        question: str,
        context: str,
        model_id: str,
        model_name: str,
    ) -> str | None:
        prompt = build_prompt(question, context)

        is_reasoning_model = any(
            reasoning_model in model_name for reasoning_model in REASONING_MODELS
        )
        max_tokens = 2048 if is_reasoning_model else 256

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 0.1,
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    message = result["choices"][0]["message"]

                    content = message.get("content", "").strip()
                    if not content and "reasoning" in message:
                        content = message["reasoning"].strip()

                    return content if content else None

                if response.status_code == 429:
                    thread_num = threading.current_thread().ident or 0
                    wait_time = (2**attempt) + (thread_num % 5)
                    time.sleep(wait_time)
                    continue

                with self.lock:
                    print(f"API Error {response.status_code} for {model_name}")
                break

            except Exception as e:
                with self.lock:
                    print(f"Request error for {model_name}: {e}")
                if attempt < 2:
                    time.sleep(1)

        return None


def process_batch_concurrent(
    client: OpenRouterClient,
    batch_data: list[tuple[int, str, str]],
    model_id: str,
    model_name: str,
) -> dict[int, str | None]:
    def process_single(item: tuple[int, str, str]) -> tuple[int, str | None]:
        idx, question, context = item
        return idx, client.query_model(question, context, model_id, model_name)

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single, item): item for item in batch_data}

        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                item = futures[future]
                print(f"Error processing item {item[0]}: {e}")
                results[item[0]] = None

    return results


def main() -> None:
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
        model_id = OPENROUTER_MODELS[model_name]
        column_name = f"{model_name}_prediction"

        if column_name not in df.columns:
            df[column_name] = None
            print(f"{model_name}: Created new column")
        else:
            print(
                f"{model_name}: Column already exists, will only process missing values",
            )

        batch_data = []
        for idx, row in df.iterrows():
            if pd.isna(row["Question"]) or pd.isna(row["Document"]):
                continue

            existing_prediction = row.get(column_name)
            if pd.notna(existing_prediction) and str(existing_prediction).strip():
                continue

            batch_data.append((idx, str(row["Question"]), str(row["Document"])))

        if not batch_data:
            print(f"{model_name}: All predictions already exist, skipping")
            continue

        print(f"{model_name}: Starting predictions for {len(batch_data)} rows...")
        start_time = time.time()

        total_batches = (len(batch_data) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(0, len(batch_data), BATCH_SIZE):
            batch = batch_data[batch_idx : batch_idx + BATCH_SIZE]
            current_batch = (batch_idx // BATCH_SIZE) + 1

            print(
                f"{model_name}: Processing batch {current_batch} / {total_batches} ({len(batch)} items)",
            )

            batch_results = process_batch_concurrent(
                client,
                batch,  # type: ignore[arg-type]
                model_id,
                model_name,
            )

            for idx, result in batch_results.items():
                df.at[idx, column_name] = result

            df.to_csv(DATA_FILE, index=False, sep="\t", encoding="utf-8")

            print(f"{model_name}: Batch {current_batch} complete")

            if current_batch < total_batches:
                time.sleep(1)

        successful = 0
        not_in_document_count = 0
        valid_predictions = 0

        for _, row in df.iterrows():
            prediction = row.get(column_name)
            document = str(row["Document"])

            if pd.notna(prediction) and str(prediction).strip():
                successful += 1
                valid_predictions += 1
                if str(prediction).strip() not in document:
                    not_in_document_count += 1

        elapsed = time.time() - start_time

        print(f"{model_name}: Complete ({successful} / {len(df)} total predictions)")
        print(
            f"{model_name}: Processed {len(batch_data)} new predictions in {elapsed:.1f} seconds",
        )
        if len(batch_data) > 0:
            print(f"{model_name}: {len(batch_data) / elapsed:.1f} requests/sec")

        if valid_predictions > 0:
            print(
                f"{model_name}: {not_in_document_count} / {valid_predictions} predictions not found in document ({not_in_document_count / valid_predictions * 100:.1f}%)",
            )
        else:
            print(f"{model_name}: No valid predictions to check")


if __name__ == "__main__":
    main()
