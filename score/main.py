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
        except:
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


def main():
    df = pd.read_csv(DATA_FILE, sep="\t", encoding="utf-8")

    ground_truth = []
    for _, row in df.iterrows():
        if pd.notna(row.get("Answer")) and str(row["Answer"]).strip() != "-":
            ground_truth.append(str(row["Answer"]))
        else:
            ground_truth.append(str(row["Document"]))

    prediction_columns = [col for col in df.columns if col.endswith("_prediction")]

    if not prediction_columns:
        print("No prediction columns found!")
        return

    print(f"Evaluating {len(prediction_columns)} models...")
    print(f"Total samples: {len(df)}")

    results = []

    for col in prediction_columns:
        model_name = col.replace("_prediction", "")
        print(f"\nEvaluating {model_name}...")

        predictions = df[col].tolist()
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
                "meteor": meteor,
                "comet": comet,
                "bert": bert,
                "chrf": chrf,
                "valid_predictions": valid_count,
                "total_samples": len(df),
            }
        )

        print(f"  BLEU: {bleu:.2f}")
        print(f"  METEOR: {meteor:.4f}")
        print(f"  COMET: {comet:.4f}")
        print(f"  BERT: {bert:.4f}")
        print(f"  CHRF: {chrf:.4f}")
        print(f"  Valid: {valid_count}/{len(df)}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("comet", ascending=False)
    results_df.to_csv("../scores.csv", index=False)


if __name__ == "__main__":
    main()
