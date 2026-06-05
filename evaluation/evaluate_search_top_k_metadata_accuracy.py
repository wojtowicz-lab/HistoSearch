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


def calculate_metadata_accuracy(df_true: pd.DataFrame, results_metadata_list: list[pd.DataFrame], eval_cols: list[str]):
    """Calculate accuracy for the predicted embeddings on a list of evaluation columns, checking if any of the top k results match the true value."""
    accs = {}

    for col in eval_cols:
        col_true = df_true[col]
        matches = []
        for i, df_pred in enumerate(results_metadata_list):
            col_pred = df_pred[col]
            matches.append((col_true.iloc[i] == col_pred).any())
        accs[col] = sum(matches) / len(matches)

    df = pd.DataFrame([[accs[col]] for col in eval_cols], columns=["accuracy"], index=eval_cols)
    return df


def main(
        index_filepath: str,
        val_embeddings_filepath: str,
        n_samples: int,
        seed: int,
        output_dir: str,
        top_k: int,
):
    """
    Sample n embeddings from each validation slide, search top k matches in training index.
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

    # Initialize query names and result indices collection
    val_embedding_names = []
    result_embedding_indices = []  # list of lists, each sublist has top_k indices for one query

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

            # Search faiss train index for top k
            indices, distances = search_faiss_multiple(slide_embeddings, index, top_k, subset=patch_indices_train)
            
            # Collect indices for each query
            for i in range(len(slide_embedding_names)):
                result_embedding_indices.append(indices[i].tolist())

    # Get query (unaugmented) and results metadata
    val_embedding_names = [name+".jpg" for name in val_embedding_names]
    query_metadata = patch_metadata_val.loc[patch_metadata_val.patch_filename.isin(val_embedding_names) & (patch_metadata_val.rotation == 0)]
    query_metadata = query_metadata.set_index("patch_filename").loc[val_embedding_names].reset_index()
    query_metadata = pd.merge(query_metadata, patient_metadata_val, how="left", left_on="patient_id", right_on="TCGA Participant Barcode")

    results_metadata_list = []
    for indices_list in result_embedding_indices:
        results_metadata = patch_metadata_train.set_index("faiss_index").reindex(indices_list).reset_index()
        results_metadata = pd.merge(results_metadata, patient_metadata_train, how="left", left_on="patient_id", right_on="TCGA Participant Barcode")
        results_metadata_list.append(results_metadata)

    # Calculate metadata statistics of the results compared to query - accuracy
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

    metrics = calculate_metadata_accuracy(query_metadata, results_metadata_list, cols_of_interest)

    # Save metrics dataframe
    metrics.to_csv(os.path.join(output_dir, f"search_top_{top_k}_metadata_accuracy.csv"))
        

if __name__ == "__main__":

    # Arguments
    INDEX_FILEPATH = "faiss_search/faiss_indices/uni2h_IndexIVFPQ_augmented.faiss"
    VAL_EMBEDDINGS_FILEPATH = "evaluation/embeddings/uni2h_embeddings_original_val.hdf5"

    N_SAMPLES = 5      # Number of tiles to sample from each slide
    SEED = 0           # Random seed to sample tiles from
    TOP_K = 5          # Top k for search

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
        "-k", "--top-k",
        type=int,
        default=TOP_K,
        help="Top k results to retrieve for each query"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to store output results"
    )

    args = parser.parse_args()

    main(**vars(args))