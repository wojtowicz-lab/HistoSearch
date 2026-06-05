import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["top1_is_original"] = df["top1_is_original"].astype(bool)
    df["top1_same_patient"] = df["top1_same_patient"].astype(bool)
    df["top1_same_slide"] = df["top1_same_slide"].astype(bool)

    grouped = df.groupby("rotation_angle", as_index=False).agg(
        original_accuracy=("top1_is_original", "mean"),
        original_accuracy_std=("top1_is_original", "std"),
        same_patient_accuracy=("top1_same_patient", "mean"),
        same_patient_accuracy_std=("top1_same_patient", "std"),
        count=("top1_is_original", "size"),
    )

    same_slide_dist = (
        df.loc[df["top1_same_slide"], ["rotation_angle", "top1_norm_distance_to_original"]]
        .groupby("rotation_angle", as_index=False)
        .agg(
            mean_same_slide_norm_distance=("top1_norm_distance_to_original", "mean"),
            mean_same_slide_norm_distance_std=("top1_norm_distance_to_original", "std"),
        )
    )

    grouped = grouped.merge(same_slide_dist, on="rotation_angle", how="left")
    grouped = grouped.sort_values("rotation_angle")
    return grouped


def plot_metrics(stats: pd.DataFrame, output_dir: str) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        stats["rotation_angle"],
        stats["original_accuracy"],
        # yerr=stats["original_accuracy_std"],
        fmt="o-",
        capsize=4,
    )
    plt.title("Original tile top-1 accuracy vs rotation angle")
    plt.xlabel("Rotation angle (degrees)")
    plt.ylabel("Top-1 original accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original_accuracy_vs_rotation_angle.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        stats["rotation_angle"],
        stats["same_patient_accuracy"],
        # yerr=stats["same_patient_accuracy_std"],
        fmt="o-",
        capsize=4,
    )
    plt.title("Same slide top-1 accuracy vs rotation angle")
    plt.xlabel("Rotation angle (degrees)")
    plt.ylabel("Top-1 same slide accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "same_slide_accuracy_vs_rotation_angle.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        stats["rotation_angle"],
        stats["mean_same_slide_norm_distance"],
        # yerr=stats["mean_same_slide_norm_distance_std"],
        fmt="o-",
        capsize=4,
    )
    plt.title("Mean normalized same-slide distance vs rotation angle")
    plt.xlabel("Rotation angle (degrees)")
    plt.ylabel("Mean normalized distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_same_slide_norm_distance_vs_rotation_angle.png"))
    plt.close()


def main(raw_csv: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_csv)
    stats = prepare_stats(df)
    stats.to_csv(os.path.join(output_dir, "rotated_tile_similarity_statistics.csv"), index=False)
    plot_metrics(stats, output_dir)
    print(f"Saved statistics CSV and plots to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rotated tile similarity rotation statistics")
    parser.add_argument("raw_csv", type=str, help="Raw rotated tile similarity metadata CSV")
    parser.add_argument("-o", "--output-dir", type=str, default="evaluation/results/uni2h", help="Output directory for statistics and plots")
    args = parser.parse_args()
    main(args.raw_csv, args.output_dir)