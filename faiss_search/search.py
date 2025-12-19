import numpy as np
import faiss
import sqlite3
import pandas as pd
import time


def search_faiss(  # NOTE WORKS FOR BOTH IVFPQ AND FLAT INDICES
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
    # Select filtered indices
    id_selector = faiss.IDSelectorArray(subset)
    search_parameters = faiss.IVFPQSearchParameters(sel=id_selector)
    
    # Perform the similarity search with selected indices
    distances, indices = index.search(query_vector, k, params=search_parameters)

    indices = indices[0]
    distances = distances[0]

    return indices, distances


# def search_faiss_bruteforce(
#         query_vector: np.array,
#         index: faiss.IndexFlat,
#         k: int = 5,
#         subset: None | list[int] = None,
# ):
#     """
#     Brute-force FAISS search with optional subset filtering.
    
#     Args:
#         query_vector: np.array (D,) or (1, D)
#         index: faiss.IndexFlatL2 or faiss.IndexFlatIP
#         embeddings: np.array (N, D)  # same data used to build index
#         k: number of neighbors to return
#         subset: optional list of row indices to restrict search
#     """

#     # Ensure query is 2D: (1, D)
#     query = query_vector.astype("float32")
#     if query.ndim == 1:
#         query = query.reshape(1, -1)
#     elif query.shape[0] != 1:
#         query = query.reshape(1, -1)

#     # Normalize if using cosine / inner product search
#     if isinstance(index, faiss.IndexFlatIP):
#         faiss.normalize_L2(query)

#     # Select filtered indices
#     id_selector = faiss.IDSelectorArray(subset)
#     search_parameters = faiss.IVFPQSearchParameters(sel=id_selector)  # TODO FIX

#     # Full brute-force search
#     distances, indices = index.search(query, k, params=search_parameters)

#     indices = indices[0]
#     distances = distances[0]

#     return indices, distances


def get_most_similar_patches(
        query_vec: np.ndarray,
        patient_metadata: pd.DataFrame,
        patch_metadata: pd.DataFrame,
        n_patients: int = 5,
        n_patches: int = 5,
        index_filepath: str = "faiss_search/faiss_indecies/uni2h_index.faiss",
) -> pd.DataFrame:
    """Search faiss iteratively excluding top patients. User can choose number of top patients and number of patches per patient."""
    # Load pregenerated faiss index
    print("Loading faiss index...")
    index = faiss.read_index(index_filepath)

    # Normalize query vector
    query = query_vec.astype("float32").reshape(1, -1).copy()
    faiss.normalize_L2(query)

    # In case of filtering prepare a subset of indecies for prefiltering the faiss index
    subset = patch_metadata.faiss_index.to_list()

    # Initiate indices collection
    all_indices = []
    all_distances = []

    patients = set([])
    i = 0

    search_metadata = patch_metadata[["faiss_index", "patient_id"]].copy()

    for _ in range(n_patients):
        # Perform similarity search within the faiss index
        print(f"Searching faiss for patient {i+1}...")
        indices, distances = search_faiss(query, index, k=n_patches, subset=subset)
        # indices, distances = search_faiss_bruteforce(query_vec, index, k=n_patches, subset=subset)

        all_indices.extend(indices)
        all_distances.extend(distances)

        # Exclude top patients
        top_patients = set(search_metadata.loc[search_metadata.faiss_index.isin(indices), "patient_id"].unique())
        search_metadata = search_metadata.loc[-search_metadata.patient_id.isin(top_patients)]

        # Add collected patients
        for patient in top_patients:
            patients.add(patient)
            i += 1
            print(f"Found patient {i}: '{patient}'")

        # Create new subset without excluded patients
        subset = search_metadata.faiss_index.to_list()
    
        # Break when no more similar patients are found, max amount of patients was found or there are no more available patients
        if -1 in indices or len(patients) >= n_patients or len(subset) == 0:
            break
    
    print(f"Found {len(patients)} patients")

    # Convert to DataFrame
    results = pd.DataFrame({"id": all_indices, "score": all_distances})
    print("Collecting metadata...")
    
    # Join with metadata
    results = results.merge(patch_metadata, left_on="id", right_on="faiss_index", how="left")
    results = results.loc[results["id"] != -1]  # Drop unfound indecies

    # Choose top n patients
    top_n_patients = results.sort_values("score", ascending=False).patient_id.drop_duplicates().iloc[:n_patients]  # Find top n patients by IP
    results = results.loc[results["patient_id"].isin(top_n_patients)]  # Drop worst patients

    # Select only best of the augmented patch versions
    results = results.sort_values("score", ascending=False).drop_duplicates("patch_filename")

    # Merge with patient metadata
    results = results.merge(patient_metadata, left_on="patient_id", right_on="TCGA Participant Barcode", how="left")

    return results
