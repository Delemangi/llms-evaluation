from typing import cast

import pandas as pd
import torch
from prompt import build_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_FILE = "../data.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HUGGINGFACE_MODELS = {
    "mkllm_7b": "trajkovnikola/MKLLM-7B-Instruct",
}

MODELS_TO_TEST = ["mkllm_7b"]


class HuggingFaceClient:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, model_name: str):
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )

        if DEVICE == "cpu":
            self.model = self.model.to(DEVICE)

        print(f"Model loaded on {DEVICE}")

    def query_model(self, question: str, context: str) -> str | None:
        if self.model is None or self.tokenizer is None:
            return None

        prompt = build_prompt(question, context)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "КОПИРАЈ ТОЧНО ОД ДОКУМЕНТОТ:" in response:
                answer = response.split("КОПИРАЈ ТОЧНО ОД ДОКУМЕНТОТ:")[-1].strip()
            else:
                answer = response.strip()

            unwanted_tokens = [
                "<|im_end|>",
                "<|im_start|>",
                "<|assistant|>",
                "<|user|>",
                "<|system|>",
            ]
            for token in unwanted_tokens:
                answer = answer.replace(token, "")

            return answer.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return None


def main():
    df = pd.read_csv(DATA_FILE, sep="\t", encoding="utf-8")

    required_cols = ["Question", "Document"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return

    client = HuggingFaceClient()

    for model_name in MODELS_TO_TEST:
        print(f"{model_name}: Starting predictions...")

        hf_model_id = HUGGINGFACE_MODELS[model_name]
        column_name = f"{model_name}_prediction"

        client.load_model(hf_model_id)

        predictions = []
        for idx, row in df.iterrows():
            if pd.isna(row["Question"]) or pd.isna(row["Document"]):
                predictions.append(None)
                continue

            prediction = client.query_model(
                question=str(row["Question"]),
                context=str(row["Document"]),
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
