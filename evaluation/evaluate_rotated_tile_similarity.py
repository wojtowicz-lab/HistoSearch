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


torch.set_num_threads(2)  # Limit PyTorch intra-op threads to reduce CPU contention
torch.set_num_interop_threads(1)  # Limit PyTorch inter-op threads to reduce CPU contention
faiss.omp_set_num_threads(2)  # Limit FAISS to 2 threads to reduce CPU contention


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    patch_metadata = pd.read_parquet("metadata/patch_metadata.parquet")
    train_metadata = pd.read_parquet("evaluation/data/patient_metadata_train.parquet")
    val_metadata = pd.read_parquet("evaluation/data/patient_metadata_val.parquet")
    patient_metadata = pd.concat([train_metadata, val_metadata], ignore_index=True)
    patient_metadata = patient_metadata.drop_duplicates(subset=["TCGA Participant Barcode"])

    # Filter to rotation=0 only
    patch_metadata = patch_metadata.loc[patch_metadata.rotation == 0].copy()
    if patch_metadata.empty:
        raise ValueError("No rotation=0 patches found for validation patients.")

    return patch_metadata, patient_metadata


def get_neighbors(origin_row, patch_metadata: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """Get up, down, left, right neighbors. Return empty if not all four are present."""
    try:
        context_df, tile_size = get_patch_context(
            origin_row.patch_filename,
            patch_metadata,
            context_size=1,
            tile_size="auto",
            verbose=False,
            return_tile_size=True,
        )
        neighbors = {}
        for _, row in context_df.iterrows():
            x_pos = int(row["x_pos"])
            y_pos = int(row["y_pos"])
            if (x_pos, y_pos) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # left, right, down, up
                neighbors[(x_pos, y_pos)] = row
        # Require all four neighbors
        if len(neighbors) < 4:
            return None, tile_size
        return neighbors, tile_size
    except Exception:
        return None, None


def create_grid_image(center_img: Image.Image, neighbors: dict[tuple[int, int], dict], mapping_filename: str) -> Image.Image:
    """Create 3x3 grid with center and neighbors."""
    w, h = center_img.size
    grid_size = (3 * w, 3 * h)
    grid = Image.new(center_img.mode, grid_size, (0, 0, 0))  # black background

    # Paste center
    grid.paste(center_img, (w, h))

    # Paste neighbors
    pos_to_coord = {
        (-1, 0): (0, h),    # left
        (1, 0): (2*w, h),   # right
        (0, -1): (w, 0),    # down
        (0, 1): (w, 2*h),   # up
    }

    for pos, neighbor_row in neighbors.items():
        if pos in pos_to_coord:
            patch_urls = get_patch_urls([neighbor_row["patch_filename"]], mapping_filename)
            _, neighbor_img = load_patch_to_RAM(
                patch_urls[neighbor_row["patch_filename"]],
                neighbor_row["patch_filename"],
                verbose=False
            )
            grid.paste(neighbor_img, pos_to_coord[pos])

    return grid


def rotate_and_crop_center(grid_img: Image.Image, angle: float) -> Image.Image:
    """Rotate grid and crop center tile."""
    w, h = grid_img.size

    rotated = grid_img.rotate(angle, expand=True, fillcolor=(0, 0, 0))
    # Center crop: assuming rotation keeps aspect, but to be safe, crop from center
    rot_w, rot_h = rotated.size
    crop_w, crop_h = w // 3, h // 3  # size of the center tile

    left = (rot_w - crop_w) // 2
    top = (rot_h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = rotated.crop((left, top, right, bottom))
    return cropped


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
        n_rotations: int,
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

    # Rotation angles
    angles = [i * (90 / (n_rotations - 1)) for i in range(n_rotations)] if n_rotations > 1 else [0]

    for slide_name in tqdm(slide_keys, desc="Slides"):
        slide_patches = patch_metadata.loc[patch_metadata.slide == slide_name]
        
        selected = []
        for _ in range(n_tiles_per_slide * 3):
            if len(selected) >= n_tiles_per_slide:
                break
            origin_row = slide_patches.sample(n=1).iloc[0]
            neighbors, tile_size = get_neighbors(origin_row, patch_metadata)
            if not neighbors:  # Require all neighbors
                continue
            selected.append((origin_row, neighbors))
        
        for origin_row, neighbors in selected:
            # Fetch center image
            origin_patch_urls = get_patch_urls([origin_row.patch_filename], mapping_filename)
            _, center_img = load_patch_to_RAM(
                origin_patch_urls[origin_row.patch_filename],
                origin_row.patch_filename,
                verbose=False
            )

            # Create grid
            grid_img = create_grid_image(center_img, neighbors, mapping_filename)

            # Generate rotated crops
            rotated_images = []
            for angle in angles:
                rotated_crop = rotate_and_crop_center(grid_img, angle)
                rotated_images.append(rotated_crop)

            # DEBUG: Save example grid and rotated crops for first slide
            if slide_name == slide_keys[0] and origin_row.patch_filename == selected[0][0].patch_filename:
                example_dir = os.path.join(output_dir, "example_grid_and_rotations")
                os.makedirs(example_dir, exist_ok=True)
                grid_img.save(os.path.join(example_dir, f"{origin_row.patch_filename}_grid.png"))
                for angle, rot_img in zip(angles, rotated_images):
                    rot_img.save(os.path.join(example_dir, f"{origin_row.patch_filename}_rot{int(angle)}.png"))

            # Embed all rotated images in batch
            rotated_embeddings = embed_batch(rotated_images, model, transform, device, batch_size)

            # Search all rotated embeddings at once with FAISS
            query_batch = rotated_embeddings.astype("float32").copy()
            faiss.normalize_L2(query_batch)
            distances, indices = index.search(query_batch, 1, params=search_parameters)

            for rot_step, (angle, top1_index, similarity) in enumerate(zip(angles, indices[:, 0], distances[:, 0])):
                if int(top1_index) not in patch_metadata_by_index.index:
                    continue
                top1_row = patch_metadata_by_index.loc[int(top1_index)]
                same_slide = top1_row.slide == origin_row.slide
                same_patient = top1_row.patient_id == origin_row.patient_id
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
                    "rotation_angle": float(angle),
                    "rot_step": int(rot_step),
                    "top1_faiss_index": int(top1_index),
                    "top1_patch_filename": top1_row.patch_filename,
                    "top1_slide": top1_row.slide,
                    "top1_patient_id": top1_row.patient_id,
                    "top1_patch_coord_x": int(top1_row.patch_coord_x),
                    "top1_patch_coord_y": int(top1_row.patch_coord_y),
                    "top1_rotation": int(top1_row.rotation),
                    "top1_similarity": float(similarity),
                    "top1_is_original": bool(top1_index == origin_row.faiss_index),
                    "top1_same_patient": bool(same_patient),
                    "top1_same_slide": bool(same_slide),
                    "top1_norm_distance_to_original": float(norm_distance) if norm_distance is not None else None,
                })

    raw_path = os.path.join(output_dir, "rotated_tile_similarity_raw.csv")
    pd.DataFrame(results).to_csv(raw_path, index=False)
    print(f"Raw rotation similarity metadata written to: {raw_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raw rotated-tile similarity search metadata using grid rotation and UNI2-h embeddings")
    parser.add_argument("-i", "--index-filepath", type=str, default="faiss_search/faiss_indices/uni2h_IndexIVFPQ_augmented.faiss", help="FAISS index file path")
    parser.add_argument("-n", "--n-tiles-per-slide", type=int, default=1, help="Number of random tiles to sample per slide")
    parser.add_argument("-m", "--n-rotations", type=int, default=10, help="Number of rotation steps from 0 to 90 degrees")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-slides", type=int, default=None, help="Maximum number of slides to evaluate")
    parser.add_argument("-o", "--output-dir", type=str, default="evaluation/results/uni2h", help="Directory to store raw metadata results")
    parser.add_argument("--mapping-filename", type=str, default="retrieval/data/patch_name_to_drive_id_mapping.parquet", help="Patch name to Google Drive ID mapping file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference")

    args = parser.parse_args()
    main(**vars(args))