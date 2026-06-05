import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["top1_is_original"] = df["top1_is_original"].astype(bool)
    df["top1_is_neighbor"] = df["top1_is_neighbor"].astype(bool)
    df["top1_same_slide"] = df["top1_same_slide"].astype(bool)

    grouped = df.groupby("neighbor_pct", as_index=False).agg(
        original_accuracy=("top1_is_original", "mean"),
        original_accuracy_std=("top1_is_original", "std"),
        neighbor_accuracy=("top1_is_neighbor", "mean"),
        neighbor_accuracy_std=("top1_is_neighbor", "std"),
        same_slide_rate=("top1_same_slide", "mean"),
        same_slide_rate_std=("top1_same_slide", "std"),
        count=("top1_is_original", "size"),
    )

    same_slide_dist = (
        df.loc[df["top1_same_slide"], ["neighbor_pct", "top1_norm_distance_to_original"]]
        .groupby("neighbor_pct", as_index=False)
        .agg(
            mean_same_slide_norm_distance=("top1_norm_distance_to_original", "mean"),
            mean_same_slide_norm_distance_std=("top1_norm_distance_to_original", "std"),
        )
    )

    grouped = grouped.merge(same_slide_dist, on="neighbor_pct", how="left")
    grouped = grouped.sort_values("neighbor_pct")
    return grouped


def plot_metrics(stats: pd.DataFrame, output_dir: str) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    plt.plot(
        stats["neighbor_pct"],
        stats["original_accuracy"],
        "o-",
        label="Original accuracy",
    )
    plt.plot(
        stats["neighbor_pct"],
        stats["neighbor_accuracy"],
        "s--",
        label="Neighbor accuracy",
    )
    plt.title("Top-1 accuracy vs alpha shift (neighbor_pct)")
    plt.xlabel("alpha shift (neighbor_pct)")
    plt.ylabel("Top-1 accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_neighbor_pct.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        stats["neighbor_pct"],
        stats["same_slide_rate"],
        "o-",
    )
    plt.title("Top-1 same-slide rate vs alpha shift (neighbor_pct)")
    plt.xlabel("alpha shift (neighbor_pct)")
    plt.ylabel("Same-slide top-1 fraction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "same_slide_rate_vs_neighbor_pct.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        stats["neighbor_pct"],
        stats["mean_same_slide_norm_distance"],
        "o-",
    )
    plt.title("Mean normalized same-slide distance vs alpha shift (neighbor_pct)")
    plt.xlabel("alpha shift (neighbor_pct)")
    plt.ylabel("Mean normalized distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_same_slide_norm_distance_vs_neighbor_pct.png"))
    plt.close()


def main(raw_csv: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_csv)
    stats = prepare_stats(df)
    stats.to_csv(os.path.join(output_dir, "shifted_tile_similarity_statistics.csv"), index=False)
    plot_metrics(stats, output_dir)
    print(f"Saved statistics CSV and plots to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot shifted tile similarity shift statistics")
    parser.add_argument("raw_csv", type=str, help="Raw shifted tile similarity metadata CSV")
    parser.add_argument("-o", "--output-dir", type=str, default="evaluation/results/uni2h", help="Output directory for statistics and plots")
    args = parser.parse_args()
    main(args.raw_csv, args.output_dir)
