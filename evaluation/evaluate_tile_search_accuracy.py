import argparse
import os
import random
import faiss
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from PIL import Image

from models.utils import load_UNI2h
from retrieval.fetch import load_patch_to_RAM
from retrieval.utils import get_patch_urls

# CPU limits
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
faiss.omp_set_num_threads(1)


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    patch_metadata = pd.read_parquet("metadata/patch_metadata.parquet")
    train_metadata = pd.read_parquet("evaluation/data/patient_metadata_train.parquet")
    val_metadata = pd.read_parquet("evaluation/data/patient_metadata_val.parquet")
    patient_metadata = pd.concat([train_metadata, val_metadata], ignore_index=True)
    patient_metadata = patient_metadata.drop_duplicates(subset=["TCGA Participant Barcode"])

    # Filter to rotation=0 only
    patch_metadata = patch_metadata.loc[patch_metadata.rotation == 0].copy()
    if patch_metadata.empty:
        raise ValueError("No rotation=0 patches found.")

    return patch_metadata, patient_metadata, train_metadata, val_metadata


def main(
        index_filepath: str,
        n_tiles_per_slide: int,
        seed: int,
        output_dir: str,
        max_slides: int | None,
        mapping_filename: str,
        batch_size: int,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    patch_metadata, patient_metadata, train_metadata, val_metadata = load_metadata()
    patch_metadata_by_index = patch_metadata.set_index("faiss_index")

    index = faiss.read_index(index_filepath)

    # FAISS subset for rotation=0 validation patches
    subset_indices = patch_metadata.faiss_index.to_numpy(dtype=np.int64)
    id_selector = faiss.IDSelectorArray(subset_indices)
    search_parameters = faiss.IVFPQSearchParameters(sel=id_selector)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, transform = load_UNI2h()
    model = model.to(device)

    results = []

    slide_keys = list(patch_metadata.loc[patch_metadata.patient_id.isin(val_metadata["TCGA Participant Barcode"])]["slide"].unique())
    random.shuffle(slide_keys)

    if max_slides is not None:
        slide_keys = slide_keys[:max_slides]

    train_patient_ids = set(train_metadata["TCGA Participant Barcode"].values)
    val_patient_ids = set(val_metadata["TCGA Participant Barcode"].values)

    for slide_name in tqdm(slide_keys, desc="Slides"):
        slide_patches = patch_metadata.loc[patch_metadata.slide == slide_name]

        # Sample n_tiles_per_slide tiles
        selected = []
        for _ in range(n_tiles_per_slide * 3):
            if len(selected) >= n_tiles_per_slide:
                break
            tile_row = slide_patches.sample(n=1).iloc[0]
            selected.append(tile_row)

        # Load images
        tile_images = []
        tile_rows = []

        for tile_row in selected:
            try:
                patch_urls = get_patch_urls([tile_row.patch_filename], mapping_filename)
                _, img = load_patch_to_RAM(
                    patch_urls[tile_row.patch_filename],
                    tile_row.patch_filename,
                    verbose=False
                )
                tile_images.append(img)
                tile_rows.append(tile_row)
            except Exception:
                continue

        if not tile_images:
            continue

        # Embed all tiles in one batch
        batch_tensors = torch.stack([transform(img) for img in tile_images]).to(device)
        with torch.no_grad():
            embeddings = model(batch_tensors)
        embeddings = embeddings.cpu().numpy()

        # Search
        query_batch = embeddings.astype("float32").copy()
        faiss.normalize_L2(query_batch)
        distances, indices = index.search(query_batch, 1, params=search_parameters)

        # Record results
        for tile_row, top1_index, similarity in zip(tile_rows, indices[:, 0], distances[:, 0]):
            if int(top1_index) not in patch_metadata_by_index.index:
                continue

            top1_row = patch_metadata_by_index.loc[int(top1_index)]

            # Get patient metadata
            patient_id = tile_row.patient_id
            patient_info = patient_metadata.loc[
                patient_metadata["TCGA Participant Barcode"] == patient_id
            ]

            anatomic_origin = "Unknown"
            stage = "Unknown"
            if not patient_info.empty:
                anatomic_origin = str(patient_info.iloc[0].get("Organ", "Unknown"))
                stage = str(patient_info.iloc[0].get("Pathologic Stage", "Unknown"))

            cohort = patient_info.iloc[0].get("TCGA Project Code", "Unknown") if not patient_info.empty else "Unknown"

            results.append({
                "query_slide": tile_row.slide,
                "query_patch_filename": tile_row.patch_filename,
                "query_patient_id": tile_row.patient_id,
                "query_faiss_index": int(tile_row.faiss_index),
                "top1_faiss_index": int(top1_index),
                "top1_patch_filename": top1_row.patch_filename,
                "top1_slide": top1_row.slide,
                "top1_patient_id": top1_row.patient_id,
                "top1_similarity": float(similarity),
                "is_original": bool(int(top1_index) == tile_row.faiss_index),
                "same_patient": bool(top1_row.patient_id == tile_row.patient_id),
                "same_slide": bool(top1_row.slide == tile_row.slide),
                "anatomic_origin": anatomic_origin,
                "stage": stage,
                "cohort": cohort,
            })

    raw_path = os.path.join(output_dir, "tile_search_accuracy_raw.csv")
    pd.DataFrame(results).to_csv(raw_path, index=False)
    print(f"Raw tile search accuracy metadata written to: {raw_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tile search accuracy metadata using direct FAISS search")
    parser.add_argument("-i", "--index-filepath", type=str, default="faiss_search/faiss_indices/uni2h_IndexIVFPQ_augmented.faiss", help="FAISS index file path")
    parser.add_argument("-n", "--n-tiles-per-slide", type=int, default=10, help="Number of random tiles to sample per slide")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-slides", type=int, default=None, help="Maximum number of slides to evaluate")
    parser.add_argument("-o", "--output-dir", type=str, default="evaluation/results/uni2h", help="Directory to store raw metadata results")
    parser.add_argument("--mapping-filename", type=str, default="retrieval/data/patch_name_to_drive_id_mapping.parquet", help="Patch name to Google Drive ID mapping file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference (unused, all tiles per slide in one batch)")

    args = parser.parse_args()
    main(**vars(args))
