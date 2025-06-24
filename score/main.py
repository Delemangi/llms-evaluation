from collections import Counter

import nltk
import numpy as np
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from comet.models import CometModel
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

COMET_MODEL: CometModel | None = None
COMET_CACHE: dict[str, float] = {}


def load_comet_model() -> CometModel | None:
    global COMET_MODEL  # noqa: PLW0603

    if COMET_MODEL is None:
        try:
            print("Loading COMET model (this may take a moment)...")
            model_path = download_model("Unbabel/wmt22-comet-da")
            COMET_MODEL = load_from_checkpoint(model_path)
            print("COMET model loaded successfully!")
        except Exception as e:
            print(f"Error loading COMET model: {e}")
            COMET_MODEL = None

    return COMET_MODEL


def calculate_all_comet_scores(
    all_evaluation_data: dict[str, tuple[list[str], list[str], list[str]]],
) -> dict[str, float]:
    global COMET_CACHE  # noqa: PLW0602

    if COMET_MODEL is None:
        print("COMET model not available")
        return {}

    if not all_evaluation_data:
        return {}

    try:
        print(f"Computing COMET scores for {len(all_evaluation_data)} evaluations...")
        use_gpu = torch.cuda.is_available()

        batch_data = []
        for sources, references, predictions in all_evaluation_data.values():
            valid_data = []
            for src, ref, pred in zip(sources, references, predictions, strict=True):
                if pd.notna(src) and pd.notna(ref) and pd.notna(pred):
                    valid_data.append(
                        {
                            "src": str(src).strip(),
                            "ref": str(ref).strip(),
                            "mt": str(pred).strip(),
                        },
                    )
            batch_data.extend(valid_data)

        if not batch_data:
            return {}

        model_output = COMET_MODEL.predict(
            batch_data,
            batch_size=32 if use_gpu else 16,
            gpus=1 if use_gpu else 0,
            progress_bar=True,
        )

        scores = model_output.scores
        score_idx = 0

        for eval_key, (sources, references, predictions) in all_evaluation_data.items():
            valid_scores = []

            for src, ref, pred in zip(sources, references, predictions, strict=True):
                if (
                    pd.notna(src)
                    and pd.notna(ref)
                    and pd.notna(pred)
                    and score_idx < len(scores)
                ):
                    valid_scores.append(scores[score_idx])
                    score_idx += 1

            if valid_scores:
                COMET_CACHE[eval_key] = float(np.mean(valid_scores))
            else:
                COMET_CACHE[eval_key] = 0.0

        print("COMET scores computed successfully!")
        return COMET_CACHE  # noqa: TRY300

    except Exception as e:
        print(f"Error calculating COMET batch: {e}")
        return {}


def calculate_bleu_score(references: list[str], predictions: list[str]) -> float:
    valid_pairs = []
    for ref, pred in zip(references, predictions, strict=True):
        if pd.notna(ref) and pd.notna(pred):
            valid_pairs.append((str(ref).strip(), str(pred).strip()))

    if not valid_pairs:
        return 0.0

    refs, preds = zip(*valid_pairs, strict=True)
    bleu = BLEU()
    return bleu.corpus_score(preds, [refs]).score


def calculate_meteor_score(references: list[str], predictions: list[str]) -> float:
    valid_pairs = []
    for ref, pred in zip(references, predictions, strict=True):
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

    return float(np.mean(meteor_scores)) if meteor_scores else 0.0


def calculate_chrf_score(references: list[str], predictions: list[str]) -> float:
    valid_pairs = []
    for ref, pred in zip(references, predictions, strict=True):
        if pd.notna(ref) and pd.notna(pred):
            valid_pairs.append((str(ref).strip(), str(pred).strip()))

    if not valid_pairs:
        return 0.0

    refs, preds = zip(*valid_pairs, strict=True)
    chrf = CHRF()
    return chrf.corpus_score(preds, [refs]).score


def get_consensus_ground_truth(
    df: pd.DataFrame,
    prediction_columns: list[str],
) -> tuple[list[str], int, int]:
    ground_truth = []
    consensus_count = 0
    original_count = 0

    for _, row in df.iterrows():
        real_predictions = []
        for col in prediction_columns:
            pred = row[col]
            if pd.notna(pred) and str(pred).strip():
                pred_clean = str(pred).strip()
                if pred_clean not in ["Нема.", "Нема", ""]:
                    real_predictions.append(pred_clean)

        consensus_found = False
        if real_predictions:
            pred_counts = Counter(real_predictions)
            total_real_predictions = len(real_predictions)
            threshold = max(total_real_predictions * 0.33, 4)

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


def filter_by_dialect(
    df: pd.DataFrame,
    dialect_value: int,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if "Dialect" not in df.columns:
        return df, [], [], []

    filtered_df = df[df["Dialect"] == dialect_value]
    if filtered_df.empty:
        return filtered_df, [], [], []

    sources = filtered_df["Question"].tolist()

    prediction_columns = [col for col in df.columns if col.endswith("_prediction")]
    ground_truth, _, _ = get_consensus_ground_truth(filtered_df, prediction_columns)

    return filtered_df, sources, ground_truth, prediction_columns


def prepare_evaluation_data(
    df: pd.DataFrame,
    prediction_columns: list[str],
    has_dialect: bool = False,
) -> dict[str, tuple[list[str], list[str], list[str]]]:
    all_evaluation_data = {}

    overall_ground_truth, _, _ = get_consensus_ground_truth(df, prediction_columns)
    overall_sources = df["Question"].tolist()

    for col in prediction_columns:
        model_name = col.replace("_prediction", "")
        predictions = preprocess_predictions(df[col].tolist())

        eval_key = f"{model_name}_overall"
        all_evaluation_data[eval_key] = (
            overall_sources,
            overall_ground_truth,
            predictions,
        )

        if has_dialect:
            for dialect_val in [0, 1]:
                filtered_df, sources, ground_truth, _ = filter_by_dialect(
                    df,
                    dialect_val,
                )
                if not filtered_df.empty:
                    filtered_predictions = preprocess_predictions(
                        filtered_df[col].tolist(),
                    )
                    eval_key = f"{model_name}_dialect_{dialect_val}"
                    all_evaluation_data[eval_key] = (
                        sources,
                        ground_truth,
                        filtered_predictions,
                    )

    return all_evaluation_data


def evaluate_model_by_dialect(
    df: pd.DataFrame,
    col: str,
    dialect_value: int | None = None,
) -> dict[str, float] | None:
    model_name = col.replace("_prediction", "")

    if dialect_value is not None:
        eval_key = f"{model_name}_dialect_{dialect_value}"
        filtered_df, _, ground_truth, _ = filter_by_dialect(df, dialect_value)
        if filtered_df.empty:
            return None
        predictions = filtered_df[col].tolist()
        filtered_df_for_count = filtered_df
    else:
        eval_key = f"{model_name}_overall"
        prediction_columns = [col for col in df.columns if col.endswith("_prediction")]
        ground_truth, _, _ = get_consensus_ground_truth(df, prediction_columns)
        predictions = df[col].tolist()
        filtered_df_for_count = df

    predictions = preprocess_predictions(predictions)

    bleu = calculate_bleu_score(ground_truth, predictions)
    chrf = calculate_chrf_score(ground_truth, predictions)
    meteor = calculate_meteor_score(ground_truth, predictions)

    comet = COMET_CACHE.get(eval_key, 0.0)

    valid_count = sum(1 for p in predictions if pd.notna(p))

    return {
        "bleu": bleu,
        "chrf": chrf,
        "meteor": meteor,
        "comet": comet,
        "valid_predictions": valid_count,
        "total_samples": len(filtered_df_for_count),
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE, sep="\t", encoding="utf-8")

    prediction_columns = [col for col in df.columns if col.endswith("_prediction")]

    if not prediction_columns:
        print("No prediction columns found!")
        return

    comet_model = load_comet_model()

    has_dialect = "Dialect" in df.columns
    if has_dialect:
        unique_dialects = sorted(df["Dialect"].unique())
        print(f"Found dialect values: {unique_dialects}")
        dialect_0_count = len(df[df["Dialect"] == 0])
        dialect_1_count = len(df[df["Dialect"] == 1])
        print(f"Dialect 0 samples: {dialect_0_count}")
        print(f"Dialect 1 samples: {dialect_1_count}")
    else:
        print("No Dialect column found - showing overall results only")

    print(f"Evaluating {len(prediction_columns)} models...")
    print(f"Total samples: {len(df)}")

    _, consensus_count, original_count = get_consensus_ground_truth(
        df,
        prediction_columns,
    )

    print("\nGround Truth Sources:")
    print(
        f"  Consensus (≥33% models agree): {consensus_count} / {len(df)} ({consensus_count / len(df) * 100:.1f}%)",
    )
    print(
        f"  Original (Answer/Document):    {original_count} / {len(df)} ({original_count / len(df) * 100:.1f}%)",
    )

    all_evaluation_data = prepare_evaluation_data(df, prediction_columns, has_dialect)
    if comet_model is not None:
        calculate_all_comet_scores(all_evaluation_data)

    all_results = []

    for col in prediction_columns:
        model_name = col.replace("_prediction", "")
        print(f"\nEvaluating {model_name}...")

        overall_results = evaluate_model_by_dialect(df, col)
        if overall_results:
            overall_results["model"] = model_name
            overall_results["dialect"] = "overall"
            all_results.append(overall_results)

            print("  Overall Results:")
            print(f"    BLEU:      {overall_results['bleu']:.2f}")
            print(f"    chrF:      {overall_results['chrf']:.2f}")
            print(f"    METEOR:    {overall_results['meteor']:.4f}")
            print(f"    COMET:     {overall_results['comet']:.4f}")
            print(
                f"    Valid:     {overall_results['valid_predictions']}/{overall_results['total_samples']}",
            )

        # Dialect-specific evaluation
        if has_dialect:
            for dialect_val in [0, 1]:
                dialect_results = evaluate_model_by_dialect(df, col, dialect_val)
                if dialect_results:
                    dialect_results["model"] = model_name
                    dialect_results["dialect"] = f"dialect_{dialect_val}"
                    all_results.append(dialect_results)

                    print(f"  Dialect {dialect_val} Results:")
                    print(f"    BLEU:      {dialect_results['bleu']:.2f}")
                    print(f"    chrF:      {dialect_results['chrf']:.2f}")
                    print(f"    METEOR:    {dialect_results['meteor']:.4f}")
                    print(f"    COMET:     {dialect_results['comet']:.4f}")
                    print(
                        f"    Valid:     {dialect_results['valid_predictions']}/{dialect_results['total_samples']}",
                    )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv("../detailed_scores.csv", index=False)
    print("\nDetailed results saved to ../detailed_scores.csv")

    print(f"\n{'=' * 100}")
    print("FINAL RESULTS")
    print(f"{'=' * 100}")

    categories = ["overall"]
    if has_dialect:
        categories.extend(["dialect_0", "dialect_1"])

    for category in categories:
        category_results = results_df[results_df["dialect"] == category].sort_values(
            "comet",
            ascending=False,
        )

        if category_results.empty:
            continue

        print(f"\n{category.upper().replace('_', ' ')} RESULTS:")
        print("-" * 80)
        print(
            f"{'Model':<20} {'BLEU':<8} {'chrF':<8} {'METEOR':<9} {'COMET':<8} {'Valid':<8}",
        )
        print("-" * 80)

        for _, row in category_results.iterrows():
            print(
                f"{row['model']:<20} {row['bleu']:<8.2f} {row['chrf']:<8.2f} "
                f"{row['meteor']:<9.4f} {row['comet']:<8.4f} "
                f"{row['valid_predictions']:<8}",
            )

    overall_results = results_df[results_df["dialect"] == "overall"].sort_values(
        "comet",
        ascending=False,
    )
    if not overall_results.empty:
        best_overall = overall_results.iloc[0]
        print(f"\n{'=' * 80}")
        print("BEST MODEL SUMMARY (Overall)")
        print(f"{'=' * 80}")
        print(f"Model: {best_overall['model']}")
        print(f"COMET Score:    {best_overall['comet']:.4f}")
        print(f"chrF Score:     {best_overall['chrf']:.2f}")
        print(f"BLEU Score:     {best_overall['bleu']:.2f}")
        print(f"METEOR Score:   {best_overall['meteor']:.4f}")
        print(
            f"Valid Predictions: {best_overall['valid_predictions']}/{best_overall['total_samples']}",
        )

    if has_dialect:
        for dialect_val in [0, 1]:
            dialect_results = results_df[
                results_df["dialect"] == f"dialect_{dialect_val}"
            ].sort_values(
                "comet",
                ascending=False,
            )
            if not dialect_results.empty:
                best_dialect = dialect_results.iloc[0]
                print(f"\n{'=' * 80}")
                print(f"BEST MODEL SUMMARY (Dialect {dialect_val})")
                print(f"{'=' * 80}")
                print(f"Model: {best_dialect['model']}")
                print(f"COMET Score:    {best_dialect['comet']:.4f}")
                print(f"chrF Score:     {best_dialect['chrf']:.2f}")
                print(f"BLEU Score:     {best_dialect['bleu']:.2f}")
                print(f"METEOR Score:   {best_dialect['meteor']:.4f}")
                print(
                    f"Valid Predictions: {best_dialect['valid_predictions']}/{best_dialect['total_samples']}",
                )

    print(f"\n{'=' * 50}")
    print("METRIC LEADERS (Overall)")
    print(f"{'=' * 50}")

    overall_results = results_df[results_df["dialect"] == "overall"]
    if not overall_results.empty:
        for metric in ["comet", "chrf", "bleu", "meteor"]:
            best_for_metric = overall_results.sort_values(metric, ascending=False).iloc[
                0
            ]
            metric_name = metric.replace("_", "").upper()
            print(
                f"{metric_name:<10}: {best_for_metric['model']} ({best_for_metric[metric]:.4f})",
            )


if __name__ == "__main__":
    main()
