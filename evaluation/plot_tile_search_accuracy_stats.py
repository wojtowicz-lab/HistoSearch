import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_and_prepare_data(raw_csv_path: str) -> pd.DataFrame:
    """Load raw CSV and compute accuracy metrics."""
    df = pd.read_csv(raw_csv_path)
    
    # Ensure is_original is boolean
    df["is_original"] = df["is_original"].astype(bool)
    
    return df


def compute_overall_stats(df: pd.DataFrame) -> dict:
    """Compute overall search accuracy statistics."""
    total = len(df)
    correct = df["is_original"].sum()
    accuracy = correct / total * 100 if total > 0 else 0
    
    same_patient_found = (df["is_original"] & df["same_patient"]).sum()
    same_slide_found = (df["is_original"] & df["same_slide"]).sum()
    
    return {
        "total_tiles": total,
        "correct_searches": correct,
        "accuracy_pct": accuracy,
        "same_patient_correct": same_patient_found,
        "same_slide_correct": same_slide_found,
        "mean_similarity": df["top1_similarity"].mean(),
    }


def compute_grouped_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute accuracy statistics grouped by a column."""
    grouped = []
    
    for group_val in df[group_col].unique():
        group_df = df[df[group_col] == group_val]
        total = len(group_df)
        correct = group_df["is_original"].sum()
        accuracy = correct / total * 100 if total > 0 else 0
        
        grouped.append({
            group_col: group_val,
            "count": total,
            "accuracy": accuracy,
            "correct": correct,
            "mean_similarity": group_df["top1_similarity"].mean(),
            "std_similarity": group_df["top1_similarity"].std(),
        })
    
    return pd.DataFrame(grouped).sort_values("accuracy", ascending=False)


def plot_accuracy_by_group(df: pd.DataFrame, group_col: str, output_dir: str, title: str):
    """Generate bar plot of accuracy by group."""
    grouped_stats = compute_grouped_stats(df, group_col)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=grouped_stats,
        x=group_col,
        y="accuracy",
        palette="husl",
        order=grouped_stats[group_col].values
    )
    
    plt.title(f"Search Accuracy by {title}", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel(title, fontsize=12)
    plt.ylim([0, 100])
    
    # Add count labels on bars
    for i, (idx, row) in enumerate(grouped_stats.iterrows()):
        ax.text(
            i, row["accuracy"] + 2,
            f"n={row['count']}",
            ha="center", va="bottom", fontsize=9
        )
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"accuracy_by_{group_col.lower()}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_similarity_distribution(df: pd.DataFrame, output_dir: str):
    """Plot similarity distribution for correct vs incorrect searches."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    correct_df = df[df["is_original"]]
    incorrect_df = df[~df["is_original"]]
    
    axes[0].hist(correct_df["top1_similarity"], bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[0].set_title("Similarity When Original Found", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Top-1 Similarity")
    axes[0].set_ylabel("Count")
    axes[0].axvline(correct_df["top1_similarity"].mean(), color="darkgreen", linestyle="--", linewidth=2, label=f"Mean: {correct_df['top1_similarity'].mean():.3f}")
    axes[0].legend()
    
    axes[1].hist(incorrect_df["top1_similarity"], bins=30, alpha=0.7, color="red", edgecolor="black")
    axes[1].set_title("Similarity When Original Not Found", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Top-1 Similarity")
    axes[1].set_ylabel("Count")
    axes[1].axvline(incorrect_df["top1_similarity"].mean(), color="darkred", linestyle="--", linewidth=2, label=f"Mean: {incorrect_df['top1_similarity'].mean():.3f}")
    axes[1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "similarity_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_cohort_comparison(df: pd.DataFrame, output_dir: str):
    """Compare accuracy between train and val cohorts."""
    grouped = compute_grouped_stats(df, "cohort")
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(grouped["cohort"], grouped["accuracy"], color=["#1f77b4", "#ff7f0e"], edgecolor="black", linewidth=1.5)
    
    plt.title("Search Accuracy: Train vs Validation Cohort", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Cohort", fontsize=12)
    plt.ylim([0, 100])
    
    # Add value labels on bars
    for bar, row in zip(bars, grouped.itertuples()):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f"{height:.1f}%\n(n={row.count})",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_cohort_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main(raw_csv_path: str, output_dir: str = None):
    """Generate statistics and plots from raw tile search accuracy results."""
    if output_dir is None:
        output_dir = os.path.dirname(raw_csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {raw_csv_path}...")
    df = load_and_prepare_data(raw_csv_path)
    
    report_lines = []
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("=" * 60)
    
    # Overall statistics
    overall_stats = compute_overall_stats(df)
    overall_text = [
        f"Total tiles evaluated: {overall_stats['total_tiles']}",
        f"Correct searches (original found): {overall_stats['correct_searches']}",
        f"Overall accuracy: {overall_stats['accuracy_pct']:.2f}%",
        f"Same-patient correct: {overall_stats['same_patient_correct']}",
        f"Same-slide correct: {overall_stats['same_slide_correct']}",
        f"Mean similarity: {overall_stats['mean_similarity']:.4f}",
    ]
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    for line in overall_text:
        print(line)
        report_lines.append(line)
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Groupwise statistics
    for group_col, title in [("anatomic_origin", "Anatomic Origin"), 
                               ("stage", "Stage"),
                               ("cohort", "Cohort")]:
        if group_col in df.columns and df[group_col].nunique() > 1:
            header = f"Accuracy by {title}:"
            print(f"\n{header}")
            report_lines.append(header)
            grouped_stats = compute_grouped_stats(df, group_col)
            group_text = grouped_stats.to_string(index=False)
            print(group_text)
            report_lines.append(group_text)
            report_lines.append("")
    
    # Save summary statistics CSV
    summary_stats = {}
    summary_stats["overall"] = overall_stats["accuracy_pct"]
    
    for group_col in ["anatomic_origin", "stage", "cohort"]:
        if group_col in df.columns:
            grouped = compute_grouped_stats(df, group_col)
            for _, row in grouped.iterrows():
                key = f"{group_col}_{row[group_col]}"
                summary_stats[key] = row["accuracy"]
    
    stats_df = pd.DataFrame([summary_stats])
    stats_path = os.path.join(output_dir, "tile_search_accuracy_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved statistics to: {stats_path}")
    report_lines.append(f"Saved statistics to: {stats_path}")
    report_lines.append("")
    
    report_path = os.path.join(output_dir, "tile_search_accuracy_report.txt")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))
    print(f"Saved report to: {report_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    if "anatomic_origin" in df.columns and df["anatomic_origin"].nunique() > 1:
        plot_accuracy_by_group(df, "anatomic_origin", output_dir, "Anatomic Origin")
    
    if "stage" in df.columns and df["stage"].nunique() > 1:
        plot_accuracy_by_group(df, "stage", output_dir, "Stage")
    
    if "cohort" in df.columns:
        plot_cohort_comparison(df, output_dir)
    
    plot_similarity_distribution(df, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_tile_search_accuracy_stats.py <raw_csv_path> [output_dir]")
        sys.exit(1)
    
    raw_csv = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(raw_csv, output)
