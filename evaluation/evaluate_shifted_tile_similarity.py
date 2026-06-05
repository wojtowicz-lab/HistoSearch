import argparse
import os
import random
import math
import faiss
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from PIL import Image

from models.utils import load_UNI2h
from retrieval.fetch import load_patch_to_RAM
from retrieval.utils import get_patch_urls
from retrieval.context import get_patch_context


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    patch_metadata = pd.read_parquet("metadata/patch_metadata.parquet")
    train_metadata = pd.read_parquet("evaluation/data/patient_metadata_train.parquet")
    val_metadata = pd.read_parquet("evaluation/data/patient_metadata_val.parquet")
    patient_metadata = pd.concat([train_metadata, val_metadata], ignore_index=True)
    patient_metadata = patient_metadata.drop_duplicates(subset=["TCGA Participant Barcode"])

    # Keep only original rotation=0 tiles for this shifted-tile similarity evaluation.
    patch_metadata = patch_metadata.loc[patch_metadata.rotation == 0].copy()
    if patch_metadata.empty:
        raise ValueError("No rotation=0 patches found in patch_metadata.")

    return patch_metadata, patient_metadata


def find_adjacent_neighbor(
        origin_row,
        patch_metadata: pd.DataFrame,
) -> dict | None:
    """Find the rightmost neighbor (x_pos=1, y_pos=0) using get_patch_context."""
    try:
        context_df, tile_size = get_patch_context(
            origin_row.patch_filename,
            patch_metadata,
            context_size=1,
            tile_size="auto",
            verbose=False,
            return_tile_size=True,
        )
        
        # Filter for the rightmost patch
        right_neighbor_df = context_df.loc[
            (context_df["x_pos"] == 1) & (context_df["y_pos"] == 0)
        ]
        
        if len(right_neighbor_df) == 0:
            return None, tile_size
        
        return right_neighbor_df.iloc[0], tile_size
    except Exception:
        return None, None


def search_top1(index: faiss.Index, query_vec: np.ndarray) -> tuple[int, float]:
    query = query_vec.astype("float32").reshape(1, -1).copy()
    faiss.normalize_L2(query)
    distances, indices = index.search(query, 1)
    return int(indices[0, 0]), float(distances[0, 0])


def stitch_images(img1: Image.Image, img2: Image.Image, original_pct: float) -> Image.Image:
    """
    Stitch two adjacent tiles by positional shifting (sliding window effect).
    The first image is the origin and the second image is the right neighbor.

    original_pct=1.0: 100% from the origin's right side, 0% from the neighbor's left side
    original_pct=0.0: 0% from the origin's right side, 100% from the neighbor's left side
    """
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)
    
    width, height = img1.size
    origin_width = int(width * original_pct)
    neighbor_width = width - origin_width
    
    origin_segment = img1.crop((width - origin_width, 0, width, height)) if origin_width > 0 else Image.new(img1.mode, (0, height))
    neighbor_segment = img2.crop((0, 0, neighbor_width, height)) if neighbor_width > 0 else Image.new(img2.mode, (0, height))
    
    result = Image.new(img1.mode, (width, height))
    if origin_width > 0:
        result.paste(origin_segment, (0, 0))
    if neighbor_width > 0:
        result.paste(neighbor_segment, (origin_width, 0))
    
    return result


def embed_batch(
        images: list[Image.Image],
        model: torch.nn.Module,
        transform,
        device: torch.device,
        batch_size: int = 32,
) -> np.ndarray:
    """Embed a batch of PIL images using the model."""
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_tensors = torch.stack([transform(img) for img in batch]).to(device)
        
        with torch.no_grad():
            batch_embeddings = model(batch_tensors)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def main(
        index_filepath: str,
        n_tiles_per_slide: int,
        n_shifts: int,
        seed: int,
        output_dir: str,
        max_slides: int | None,
        mapping_filename: str,
        batch_size: int,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    val_metadata = pd.read_parquet("evaluation/data/patient_metadata_val.parquet")

    patch_metadata, patient_metadata = load_metadata()
    patch_metadata_by_filename = patch_metadata.set_index("patch_filename")
    patch_metadata_by_index = patch_metadata.set_index("faiss_index")

    index = faiss.read_index(index_filepath)

    # Keep a FAISS subset of rotation=0 indices only.
    subset_indices = patch_metadata.faiss_index.to_numpy(dtype=np.int64)
    if subset_indices.size == 0:
        raise ValueError("No FAISS indices available for rotation=0 patches.")
    id_selector = faiss.IDSelectorArray(subset_indices)
    search_parameters = faiss.IVFPQSearchParameters(sel=id_selector)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, transform = load_UNI2h()
    model = model.to(device)

    results = []
    
    # Select slides from validation set
    slide_keys = list(patch_metadata.loc[patch_metadata.patient_id.isin(val_metadata["TCGA Participant Barcode"])]["slide"].unique())
    random.shuffle(slide_keys)
    
    if max_slides is not None:
        slide_keys = slide_keys[:max_slides]

    for slide_name in tqdm(slide_keys, desc="Slides"):
        slide_patches = patch_metadata.loc[patch_metadata.slide == slide_name]
        
        selected = []
        for _ in range(n_tiles_per_slide * 3):
            if len(selected) >= n_tiles_per_slide:
                break
            origin_row = slide_patches.sample(n=1).iloc[0]
            
            neighbor_row, tile_size = find_adjacent_neighbor(origin_row, patch_metadata)
            if neighbor_row is None:
                continue
            
            selected.append((origin_row, neighbor_row))
        
        for origin_row, neighbor_row in selected:
            # Fetch original and neighbor images
            origin_patch_urls = get_patch_urls([origin_row.patch_filename], mapping_filename)
            neighbor_patch_urls = get_patch_urls([neighbor_row["patch_filename"]], mapping_filename)
            
            _, origin_img = load_patch_to_RAM(
                origin_patch_urls[origin_row.patch_filename],
                origin_row.patch_filename,
                verbose=False
            )
            _, neighbor_img = load_patch_to_RAM(
                neighbor_patch_urls[neighbor_row["patch_filename"]],
                neighbor_row["patch_filename"],
                verbose=False
            )

            # Generate stitched images (sliding window) and embeddings
            blended_images = []
            for step in range(n_shifts):
                original_pct = 1.0 - step / (n_shifts - 1) if n_shifts > 1 else 1.0
                stitched = stitch_images(origin_img, neighbor_img, original_pct)
                blended_images.append(stitched)

            # Embed all blended images in batch
            blended_embeddings = embed_batch(blended_images, model, transform, device, batch_size)

            # Search all shifted embeddings at once with FAISS, limited to rotation=0 patch indices
            query_batch = blended_embeddings.astype("float32").copy()
            faiss.normalize_L2(query_batch)
            distances, indices = index.search(query_batch, 1, params=search_parameters)

            for shift_step, (top1_index, similarity) in enumerate(zip(indices[:, 0], distances[:, 0])):
                alpha = 1.0 - shift_step / (n_shifts - 1) if n_shifts > 1 else 1.0
                beta = 1.0 - alpha

                if int(top1_index) not in patch_metadata_by_index.index:
                    continue
                top1_row = patch_metadata_by_index.loc[int(top1_index)]
                same_slide = top1_row.slide == origin_row.slide
                norm_distance = None
                if same_slide:
                    dx = float(origin_row.patch_coord_x) - float(top1_row.patch_coord_x)
                    dy = float(origin_row.patch_coord_y) - float(top1_row.patch_coord_y)
                    norm_distance = math.sqrt(dx * dx + dy * dy) / float(tile_size)

                results.append({
                    "query_slide": origin_row.slide,
                    "query_patch_filename": origin_row.patch_filename,
                    "query_patient_id": origin_row.patient_id,
                    "query_faiss_index": int(origin_row.faiss_index),
                    "neighbor_patch_filename": neighbor_row["patch_filename"],
                    "neighbor_patient_id": neighbor_row["patient_id"],
                    "neighbor_faiss_index": int(neighbor_row["faiss_index"]),
                    "original_pct": float(alpha),
                    "neighbor_pct": float(beta),
                    "shift_step": int(shift_step),
                    "top1_faiss_index": int(top1_index),
                    "top1_patch_filename": top1_row.patch_filename,
                    "top1_slide": top1_row.slide,
                    "top1_patient_id": top1_row.patient_id,
                    "top1_patch_coord_x": int(top1_row.patch_coord_x),
                    "top1_patch_coord_y": int(top1_row.patch_coord_y),
                    "top1_rotation": int(top1_row.rotation),
                    "top1_similarity": float(similarity),
                    "top1_is_original": bool(top1_index == origin_row.faiss_index),
                    "top1_is_neighbor": bool(top1_index == neighbor_row["faiss_index"]),
                    "top1_same_slide": bool(same_slide),
                    "top1_norm_distance_to_original": float(norm_distance) if norm_distance is not None else None,
                })

    raw_path = os.path.join(output_dir, "shifted_tile_similarity_raw.csv")
    pd.DataFrame(results).to_csv(raw_path, index=False)
    print(f"Raw shift similarity metadata written to: {raw_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raw shifted-tile similarity search metadata using positional stitching and UNI2-h embeddings")
    parser.add_argument("-i", "--index-filepath", type=str, default="faiss_search/faiss_indices/uni2h_IndexIVFPQ_augmented.faiss", help="FAISS index file path")
    parser.add_argument("-n", "--n-tiles-per-slide", type=int, default=1, help="Number of random tiles to sample per slide")
    parser.add_argument("-m", "--n-shifts", type=int, default=11, help="Number of intermediate positional shifts (sliding window steps)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-slides", type=int, default=None, help="Maximum number of slides to evaluate")
    parser.add_argument("-o", "--output-dir", type=str, default="evaluation/results/uni2h", help="Directory to store raw metadata results")
    parser.add_argument("--mapping-filename", type=str, default="retrieval/data/patch_name_to_drive_id_mapping.parquet", help="Patch name to Google Drive ID mapping file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference")

    args = parser.parse_args()
    main(**vars(args))
