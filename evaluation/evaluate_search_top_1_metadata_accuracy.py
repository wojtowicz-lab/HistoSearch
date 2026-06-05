import os
import argparse
import random
import h5py
import faiss
import numpy as np
import pandas as pd

from tqdm import tqdm


def get_patch_metadata(dataset: str = "train") -> pd.DataFrame:
    """Load only patch metadata of patients belonging to the train set."""
    # Load patch metadata
    patch_metadata_path = "metadata/patch_metadata.parquet"
    patch_metadata = pd.read_parquet(patch_metadata_path)

    # Load patient metadata
    if dataset == "train":
        patient_metadata_path = "evaluation/data/patient_metadata_train.parquet"
    elif dataset == "val":
        patient_metadata_path = "evaluation/data/patient_metadata_val.parquet"
    patient_metadata = pd.read_parquet(patient_metadata_path)

    patient_ids = patient_metadata["TCGA Participant Barcode"].to_list()

    # Filter  patch metadata
    patch_metadata = patch_metadata.loc[patch_metadata.patient_id.isin(patient_ids)]

    return patient_metadata, patch_metadata


def search_faiss_multiple(  # NOTE WORKS FOR BOTH IVFPQ AND FLAT INDICES
        query_vector: np.array,
        index: faiss.IndexIVFPQ,
        k: int = 5,
        subset: None | list[int] = None,
) -> tuple[list[float], list[float]]:
    """
    Search faiss index with optional prefiltering by embedding indecies.
    Args:
        query_vector: a query vector used for similarity search
        k: top k similar vectors to be returned
        subset: an optional list of embedding indices to search through (returned from metadata filtering)
    """
    # Normalize incomming vectors
    faiss.normalize_L2(query_vector)

    # Select filtered indices
    id_selector = faiss.IDSelectorArray(subset)
    search_parameters = faiss.IVFPQSearchParameters(sel=id_selector)
    
    # Perform the similarity search with selected indices
    distances, indices = index.search(query_vector, k, params=search_parameters)

    return indices, distances


def sample_n_embeddings():
    pass


def accuracy(tp, fp, tn, fn):
    return (tp+tn) / (tp+tn+fp+fn)


def precision(tp, fp):
    return tp / (tp+fp) if tp+fp > 0 else 0


def recall(tp, fn):
    return tp / (tp+fn) if tp+fn > 0 else 0


def f1_score(tp, fp, fn):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return (2 * prec * rec) / (prec + rec) if prec+rec > 0 else 0


def calculate_metadata_accuracy(df_true: pd.DataFrame, df_pred: pd.DataFrame, eval_cols: list[str]):
    """Calculate accuracy and other statistics for the predicted embeddings on a list of evaluation columns."""
    # For mutational signatures
    precs = {}
    recs = {}
    f1_scores = {}

    # For other variables
    accs = {}

    # Calculate statistics for each column
    for col in eval_cols:
        col_true = df_true[col]
        col_pred = df_pred[col]

        # Check accuracy for all columns
        accs[col] = (col_true == col_pred).sum() / len(col_true)

        # More precise measures for signatures
        if "SBS" in col:
            tp = ((col_true == 1) & (col_pred == 1)).sum()
            fp = ((col_true == 0) & (col_pred == 1)).sum()
            tn = ((col_true == 0) & (col_pred == 0)).sum()
            fn = ((col_true == 1) & (col_pred == 0)).sum()
            precs[col] = precision(tp, fp)
            recs[col] = recall(tp, fn)
            f1_scores[col] = f1_score(tp, fp, fn)

    signature_cols = [col for col in eval_cols if "SBS" in col]
    other_cols = [col for col in eval_cols if "SBS" not in col]

    # Build stats dataframes
    other_df = pd.DataFrame([[accs[col]] for col in other_cols], columns=["accuracy"], index=other_cols)
    signatures_df = pd.DataFrame(
        [[metric[col] for metric in [precs, recs, f1_scores]] for col in signature_cols],
        columns=["precision", "recall", "f1_score"],
        index=signature_cols,
    )
    signatures_df["accuracy"] = signatures_df.index.map(accs)

    return other_df, signatures_df


def main(
        index_filepath: str,
        val_embeddings_filepath: str,
        n_samples: int,
        seed: int,
        output_dir: str,
):
    """
    Sample n embeddings from each validation slide, search top match in training index.
    Then, compare resulting embeddings metadata with query embeddings metadata and save results into output_dir.
    """
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Load faiss index
    index = faiss.read_index(index_filepath)

    # Load metadata
    patient_metadata_train, patch_metadata_train = get_patch_metadata("train")
    patient_metadata_val, patch_metadata_val = get_patch_metadata("val")

    patient_ids_train = patient_metadata_train["TCGA Participant Barcode"].to_list()
    patch_indices_train = patch_metadata_train.loc[patch_metadata_train.patient_id.isin(patient_ids_train), "faiss_index"]    

    # Initialize query names and result indecies collection
    val_embedding_names = []
    result_embedding_indices = []

    # Evaluate search metadata accuracy
    with h5py.File(val_embeddings_filepath, "r") as src:

        for slide_id in tqdm(list(src.keys()), desc="Slides evaluated"):
            # Extract slide embeddings
            slide_embeddings = []
            slide_group = src[slide_id]
            slide_embedding_names = random.sample(list(slide_group.keys()), n_samples)  # Sample n_samples embeddings from slide
            val_embedding_names.extend(slide_embedding_names)

            # Collect sampled embeddings
            for embedding_name in slide_embedding_names:
                embedding = slide_group[embedding_name][()]
                slide_embeddings.append(embedding)

            slide_embeddings = np.array(slide_embeddings)

            # Normalize slide embeddings and search faiss train index
            indices, distances = search_faiss_multiple(slide_embeddings, index, 1, subset=patch_indices_train)
            result_embedding_indices.extend(indices.reshape(-1))
            

    # Get query (unaugmented) and results metadata
    val_embedding_names = [name+".jpg" for name in val_embedding_names]
    query_metadata = patch_metadata_val.loc[patch_metadata_val.patch_filename.isin(val_embedding_names) & (patch_metadata_val.rotation == 0)]
    query_metadata = query_metadata.set_index("patch_filename").loc[val_embedding_names].reset_index()
    query_metadata = pd.merge(query_metadata, patient_metadata_val, how="left", left_on="patient_id", right_on="TCGA Participant Barcode")

    results_metadata = patch_metadata_train.set_index("faiss_index").reindex(result_embedding_indices).reset_index()
    results_metadata = pd.merge(results_metadata, patient_metadata_train, how="left", left_on="patient_id", right_on="TCGA Participant Barcode")

    # Calculate metadata statistics of the results compared to query - accuracy, recall, precision, f1 score
    cols_of_interest = [
        "TCGA Project Code",
        "Organ",
        "Stage",
        "Gender",
        "Vital status",
        "Anatomic Region",
        "MSI Status",
    ]
    sig_cols = [col for col in query_metadata.columns if col.startswith("SBS")]
    cols_of_interest += sig_cols

    other_metrics, signature_metrics = calculate_metadata_accuracy(query_metadata, results_metadata, cols_of_interest)

    # Save metrics dataframes
    other_metrics.to_csv(os.path.join(output_dir, "search_top_1_metadata_accuracy.csv"))
    signature_metrics.to_csv(os.path.join(output_dir, "search_top_1_signatures_metrics.csv"))
        

if __name__ == "__main__":

    # Arguments
    INDEX_FILEPATH = "faiss_search/faiss_indices/uni2h_IndexIVFPQ_augmented.faiss"
    VAL_EMBEDDINGS_FILEPATH = "evaluation/embeddings/uni2h_embeddings_original_val.hdf5"

    N_SAMPLES = 5      # Number of tiles to sample from each slide
    SEED = 0           # Random seed to sample tiles from

    OUTPUT_DIR = "evaluation/results/uni2h"


    parser = argparse.ArgumentParser(
        description="Arguments for FAISS evaluation / tile sampling"
    )

    parser.add_argument(
        "-i", "--index-filepath",
        type=str,
        default=INDEX_FILEPATH,
        help="Path to the FAISS index file"
    )

    parser.add_argument(
        "-e", "--val-embeddings-filepath",
        type=str,
        default=VAL_EMBEDDINGS_FILEPATH,
        help="Path to validation embeddings HDF5 file"
    )

    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=5,
        help="Number of tiles to sample from each slide"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=0,
        help="Random seed for tile sampling"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to store output results"
    )

    args = parser.parse_args()

    main(**vars(args))



# NOTE estimated time for 1,600,000 vectors: ... h  (3:45 for 10,000 vectors)