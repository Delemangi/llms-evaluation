import warnings

import numpy as np
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from sacrebleu import BLEU, sentence_bleu

warnings.filterwarnings("ignore")


def check_gpu() -> bool:
    return torch.cuda.is_available()


def monitor_gpu_memory() -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")

        print(f"Successfully loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def calculate_bleu_scores(references: list[str], predictions: list[str]) -> dict:
    print("Calculating BLEU scores...")

    valid_pairs = []
    for ref, pred in zip(references, predictions):
        if pd.notna(ref) and pd.notna(pred):
            valid_pairs.append((str(ref).strip(), str(pred).strip()))

    if not valid_pairs:
        print("No valid pairs found for BLEU calculation")
        return {"corpus_bleu": 0.0, "avg_sentence_bleu": 0.0, "valid_samples": 0}

    refs, preds = zip(*valid_pairs)

    bleu = BLEU()
    corpus_bleu = bleu.corpus_score(preds, [refs]).score

    sentence_bleus = []
    for ref, pred in valid_pairs:
        try:
            sent_bleu = sentence_bleu(pred, [ref]).score
            sentence_bleus.append(sent_bleu)
        except Exception:
            sentence_bleus.append(0.0)

    avg_sentence_bleu = np.mean(sentence_bleus) if sentence_bleus else 0.0

    print(f"BLEU calculation complete: {len(valid_pairs)} valid samples")

    return {
        "corpus_bleu": corpus_bleu,
        "avg_sentence_bleu": avg_sentence_bleu,
        "valid_samples": len(valid_pairs),
    }


def calculate_comet_scores(
    sources: list[str],
    references: list[str],
    predictions: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    use_gpu: bool = True,
) -> dict:
    try:
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
    except Exception as e:
        print(f"Error loading COMET model: {e}")
        return {"comet_score": 0.0, "valid_samples": 0}

    valid_data = []
    for src, ref, pred in zip(sources, references, predictions):
        if pd.notna(src) and pd.notna(ref) and pd.notna(pred):
            valid_data.append(
                {
                    "src": str(src).strip(),
                    "ref": str(ref).strip(),
                    "mt": str(pred).strip(),
                }
            )

    if not valid_data:
        print("No valid data found for COMET calculation")
        return {"comet_score": 0.0, "valid_samples": 0}

    gpu_available = torch.cuda.is_available()
    use_gpu = use_gpu and gpu_available

    if use_gpu:
        gpu_count = 1
        batch_size = min(32, len(valid_data))
        monitor_gpu_memory()
    else:
        gpu_count = 0
        batch_size = 8

    print(f"Batch size: {batch_size}, Total samples: {len(valid_data)}")

    try:
        model_output = model.predict(
            valid_data,
            batch_size=batch_size,
            gpus=gpu_count,
            progress_bar=True,
        )

        if use_gpu:
            monitor_gpu_memory()

        print(f"COMET calculation complete")

        return {
            "comet_score": model_output.system_score,
            "valid_samples": len(valid_data),
        }

    except Exception as e:
        print(f"Error during COMET calculation: {e}")
        return {"comet_score": 0.0, "valid_samples": 0}


def evaluate_models(df: pd.DataFrame, use_gpu: bool = True) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STARTING MODEL EVALUATION")
    print("=" * 60)

    gpu_available = check_gpu()
    use_gpu = use_gpu and gpu_available

    model_columns = [
        "predicted_answer_ollama_prediction",
        "predicted_answer_openai_prediction",
        "predicted_answer_mkllm_prediction",
    ]

    existing_columns = [col for col in model_columns if col in df.columns]
    if not existing_columns:
        print("Error: No prediction columns found!")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    print(f"Found {len(existing_columns)} model columns to evaluate")

    results = []

    for i, model_col in enumerate(existing_columns):
        model_name = model_col.replace("predicted_answer_", "").replace(
            "_prediction", ""
        )

        print(f"\n{'-' * 40}")
        print(f"Evaluating Model {i + 1}/{len(existing_columns)}: {model_name.upper()}")
        print(f"{'-' * 40}")

        print("1. BLEU Score Calculation")
        bleu_results = calculate_bleu_scores(
            df["ground_truth_answer"].tolist(), df[model_col].tolist()
        )

        print("2. COMET Score Calculation")
        comet_results = calculate_comet_scores(
            df["question"].tolist(),
            df["ground_truth_answer"].tolist(),
            df[model_col].tolist(),
            use_gpu=use_gpu,
        )

        result = {
            "model": model_name,
            "corpus_bleu": bleu_results["corpus_bleu"],
            "avg_sentence_bleu": bleu_results["avg_sentence_bleu"],
            "comet_score": comet_results["comet_score"],
            "valid_samples": bleu_results["valid_samples"],
        }

        results.append(result)

        print(f"  BLEU: {result['corpus_bleu']:.2f}")
        print(f"  COMET: {result['comet_score']:.4f}")

    return pd.DataFrame(results)


def print_results(results_df: pd.DataFrame):
    if results_df.empty:
        print("No results to display")
        return

    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    results_sorted = results_df.sort_values("comet_score", ascending=False)

    print(
        f"{'Rank':<5} {'Model':<15} {'BLEU':<8} {'Sent-BLEU':<10} {'COMET':<8} {'Samples':<8}"
    )
    print("-" * 70)

    for rank, (_, row) in enumerate(results_sorted.iterrows(), 1):
        print(
            f"{rank:<5} {row['model']:<15} {row['corpus_bleu']:<8.2f} "
            f"{row['avg_sentence_bleu']:<10.2f} {row['comet_score']:<8.4f} "
            f"{row['valid_samples']:<8}"
        )

    print("\n" + "=" * 70)
    best_model = results_sorted.iloc[0]
    print(f"   BEST MODEL: {best_model['model'].upper()}")
    print(f"   COMET Score: {best_model['comet_score']:.4f}")
    print(f"   BLEU Score: {best_model['corpus_bleu']:.2f}")
    print("=" * 70)

    print("\n  Stats:")
    print(
        f"• Highest COMET: {results_sorted.iloc[0]['model']} ({results_sorted.iloc[0]['comet_score']:.4f})"
    )
    print(
        f"• Highest BLEU: {results_sorted.sort_values('corpus_bleu', ascending=False).iloc[0]['model']} "
        f"({results_sorted.sort_values('corpus_bleu', ascending=False).iloc[0]['corpus_bleu']:.2f})"
    )

    comet_range = (
        results_sorted["comet_score"].max() - results_sorted["comet_score"].min()
    )
    bleu_range = (
        results_sorted["corpus_bleu"].max() - results_sorted["corpus_bleu"].min()
    )
    print(f"• COMET score range: {comet_range:.4f}")
    print(f"• BLEU score range: {bleu_range:.2f}")


def save_detailed_results(
    results_df: pd.DataFrame,
    filename: str = "../score.csv",
):
    if results_df.empty:
        return

    results_with_ranks = results_df.copy()
    results_with_ranks["comet_rank"] = results_df["comet_score"].rank(
        method="dense",
        ascending=False,
    )
    results_with_ranks["bleu_rank"] = results_df["corpus_bleu"].rank(
        method="dense",
        ascending=False,
    )

    results_with_ranks.to_csv(filename, index=False)

    return results_with_ranks


def main(data_file_path: str, use_gpu: bool = True):
    df = load_data(data_file_path)
    required_columns = ["question", "ground_truth_answer"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return None

    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    pred_columns = [col for col in df.columns if "predicted_answer" in col]
    for col in pred_columns:
        non_null_count = df[col].notna().sum()
        print(f"  {col}: {non_null_count}/{len(df)} valid predictions")

    results = evaluate_models(df, use_gpu=use_gpu)
    if not results.empty:
        print_results(results)
        detailed_results = save_detailed_results(results)
        return detailed_results
    else:
        return None


if __name__ == "__main__":
    results = main("../output.csv")
