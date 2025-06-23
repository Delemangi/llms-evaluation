from collections import Counter

import nltk
import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score
from comet import download_model, load_from_checkpoint
from nltk.translate.meteor_score import meteor_score
from sacrebleu import BLEU, CHRF

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

DATA_FILE = "../data.csv"


def calculate_bleu_score(
    references: list[str],
    predictions: list[str],
) -> float:
    valid_pairs = []
    for ref, pred in zip(references, predictions):
        if pd.notna(ref) and pd.notna(pred):
            valid_pairs.append((str(ref).strip(), str(pred).strip()))

    if not valid_pairs:
        return 0.0

    refs, preds = zip(*valid_pairs)
    bleu = BLEU()
    return bleu.corpus_score(preds, [refs]).score


def calculate_meteor_score(
    references: list[str],
    predictions: list[str],
) -> float:
    valid_pairs = []
    for ref, pred in zip(references, predictions):
        if pd.notna(ref) and pd.notna(pred):
            valid_pairs.append((str(ref).strip(), str(pred).strip()))

    if not valid_pairs:
        return 0.0

    meteor_scores = []
    for ref, pred in valid_pairs:
        try:
            score = meteor_score([ref.split()], pred.split())
            meteor_scores.append(score)
        except Exception:
            meteor_scores.append(0.0)

    return np.mean(meteor_scores) if meteor_scores else 0.0


def calculate_comet_score(
    sources: list[str],
    references: list[str],
    predictions: list[str],
) -> float:
    try:
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
    except Exception as e:
        print(f"Error loading COMET model: {e}")
        return 0.0

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
        return 0.0

    try:
        use_gpu = torch.cuda.is_available()
        model_output = model.predict(
            valid_data,
            batch_size=16 if use_gpu else 8,
            gpus=1 if use_gpu else 0,
            progress_bar=False,
        )
        return model_output.system_score
    except Exception as e:
        print(f"Error calculating COMET: {e}")
        return 0.0


def calculate_chrf_score(
    references: list[str],
    predictions: list[str],
) -> float:
    valid_pairs = []
    for ref, pred in zip(references, predictions):
        if pd.notna(ref) and pd.notna(pred):
            valid_pairs.append((str(ref).strip(), str(pred).strip()))

    if not valid_pairs:
        return 0.0

    refs, preds = zip(*valid_pairs)
    chrf = CHRF()
    return chrf.corpus_score(preds, [refs]).score


def calculate_bert_score(
    references: list[str],
    predictions: list[str],
) -> float:
    valid_pairs = []
    for ref, pred in zip(references, predictions):
        if pd.notna(ref) and pd.notna(pred):
            valid_pairs.append((str(ref).strip(), str(pred).strip()))

    if not valid_pairs:
        return 0.0

    refs, preds = zip(*valid_pairs)

    P, R, F1 = bert_score(
        preds,
        refs,
        lang="mk",
        verbose=False,
        model_type="microsoft/mdeberta-v3-base",
    )
    return F1.mean().item()


def get_consensus_ground_truth(df, prediction_columns) -> tuple[list[str], int, int]:
    ground_truth = []
    consensus_count = 0
    original_count = 0

    for idx, row in df.iterrows():
        predictions = []
        for col in prediction_columns:
            pred = row[col]
            if pd.notna(pred) and str(pred).strip():
                predictions.append(str(pred).strip())

        consensus_found = False
        if predictions:
            pred_counts = Counter(predictions)
            total_predictions = len(predictions)
            threshold = max(total_predictions // 2, 5)

            for pred, count in pred_counts.most_common():
                if count >= threshold:
                    ground_truth.append(pred)
                    consensus_count += 1
                    consensus_found = True
                    break

        if not consensus_found:
            if pd.notna(row.get("Answer")) and str(row["Answer"]).strip() != "-":
                answer = str(row["Answer"]).replace("\\n", "\n")
                ground_truth.append(answer)
            else:
                ground_truth.append(str(row["Document"]))
            original_count += 1

    return ground_truth, consensus_count, original_count


def preprocess_predictions(predictions: list[str]) -> list[str]:
    return [
        "Нема." if pd.isna(p) or not str(p).strip() else str(p) for p in predictions
    ]


def main():
    df = pd.read_csv(DATA_FILE, sep="\t", encoding="utf-8")

    prediction_columns = [col for col in df.columns if col.endswith("_prediction")]

    if not prediction_columns:
        print("No prediction columns found!")
        return

    print(f"Evaluating {len(prediction_columns)} models...")
    print(f"Total samples: {len(df)}")

    ground_truth, consensus_count, original_count = get_consensus_ground_truth(
        df,
        prediction_columns,
    )

    print("\nGround Truth Sources:")
    print(
        f"  Consensus (≥50% models agree): {consensus_count} / {len(df)} ({consensus_count / len(df) * 100:.1f}%)"
    )
    print(
        f"  Original (Answer/Document):    {original_count} / {len(df)} ({original_count / len(df) * 100:.1f}%)"
    )

    results = []

    for col in prediction_columns:
        model_name = col.replace("_prediction", "")
        print(f"\nEvaluating {model_name}...")

        predictions = df[col].tolist()
        predictions = preprocess_predictions(predictions)
        sources = df["Question"].tolist()

        bleu = calculate_bleu_score(ground_truth, predictions)
        chrf = calculate_chrf_score(ground_truth, predictions)
        meteor = calculate_meteor_score(ground_truth, predictions)
        bert = calculate_bert_score(ground_truth, predictions)
        comet = calculate_comet_score(sources, ground_truth, predictions)

        valid_count = sum(1 for p in predictions if pd.notna(p))

        results.append(
            {
                "model": model_name,
                "bleu": bleu,
                "chrf": chrf,
                "meteor": meteor,
                "bert_f1": bert,
                "comet": comet,
                "valid_predictions": valid_count,
                "total_samples": len(df),
            }
        )

        print(f"  BLEU:     {bleu:.2f}")
        print(f"  chrF:     {chrf:.2f}")
        print(f"  METEOR:   {meteor:.4f}")
        print(f"  BERTScore: {bert:.4f}")
        print(f"  COMET:    {comet:.4f}")
        print(f"  Valid:    {valid_count}/{len(df)}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("comet", ascending=False)

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS (Using Consensus Ground Truth)")
    print(f"{'=' * 80}")
    print(
        f"{'Model':<20} {'BLEU':<8} {'chrF':<8} {'METEOR':<9} {'BERTScore':<10} {'COMET':<8} {'Valid':<8}"
    )
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(
            f"{row['model']:<20} {row['bleu']:<8.2f} {row['chrf']:<8.2f} "
            f"{row['meteor']:<9.4f} {row['bert_f1']:<10.4f} {row['comet']:<8.4f} "
            f"{row['valid_predictions']:<8}"
        )

    results_df["consensus_samples"] = consensus_count
    results_df["original_samples"] = original_count
    results_df.to_csv("../scores.csv", index=False)
    print("\nResults saved to ../scores.csv")

    best_model = results_df.iloc[0]
    print(f"\n{'=' * 80}")
    print("BEST MODEL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Model: {best_model['model']}")
    print(f"COMET Score:    {best_model['comet']:.4f}")
    print(f"BERTScore:      {best_model['bert_f1']:.4f}")
    print(f"chrF Score:     {best_model['chrf']:.2f}")
    print(f"BLEU Score:     {best_model['bleu']:.2f}")
    print(f"METEOR Score:   {best_model['meteor']:.4f}")
    print(
        f"Valid Predictions: {best_model['valid_predictions']}/{best_model['total_samples']}"
    )

    print("\nGround Truth Breakdown:")
    print(f"  {consensus_count} samples used consensus ground truth")
    print(f"  {original_count} samples used original ground truth")

    print(f"\n{'=' * 50}")
    print("METRIC LEADERS")
    print(f"{'=' * 50}")

    for metric in ["comet", "bert_f1", "chrf", "bleu", "meteor"]:
        best_for_metric = results_df.sort_values(metric, ascending=False).iloc[0]
        metric_name = metric.replace("_", "").upper()
        print(
            f"{metric_name:<10}: {best_for_metric['model']} ({best_for_metric[metric]:.4f})"
        )


if __name__ == "__main__":
    main()
