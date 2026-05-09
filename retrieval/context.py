import re
import os
import pandas as pd
import numpy as np

from retrieval.fetch import download_patches, load_patches_to_RAM, load_patch_to_RAM
from retrieval.utils import get_patch_urls, determine_tile_size
from PIL import Image


def get_patch_context(
        patch_filename: str,
        patch_metadata: pd.DataFrame,
        context_size: int = 1,
        tile_size: int | str = 1014,
        verbose: bool = True,
        return_tile_size: bool = False,
) -> list[str]:
    """Get nearest context images metadata of a given patch."""
    if verbose:
        print(f"Getting context of {patch_filename}...")
    x, y = re.search(r'\((\d+),(\d+)\)', patch_filename).groups()
    x, y = int(x), int(y)

    slide = patch_filename.split("_")[0]
    
    # Find nearest images in a square going `context_size` (int) patches in each direction from the given patch
    context_df = patch_metadata.copy().loc[patch_metadata.slide == slide]
    context_df["x_diff"] = (x - context_df["patch_coord_x"]).abs()
    context_df["y_diff"] = (y - context_df["patch_coord_y"]).abs()
    context_df["xy_dist"] = (context_df["x_diff"] ** 2 + context_df["y_diff"] ** 2) ** 1/2

    context_df.sort_values("xy_dist", inplace=True)

    k = (2*context_size + 1) ** 2 
    top_k_nearest = context_df.copy().iloc[:k]

    if tile_size == "auto":
        tile_size = determine_tile_size(top_k_nearest, verbose=verbose)

    # Define positions relative to the center (given patch)
    top_k_nearest["x_pos"] = ((top_k_nearest["patch_coord_x"] - x) / tile_size).apply(np.round)
    top_k_nearest["y_pos"] = ((top_k_nearest["patch_coord_y"] - y) / tile_size).apply(np.round)

    top_k_nearest = top_k_nearest.loc[(top_k_nearest.x_pos.abs() <= context_size) & (top_k_nearest.y_pos.abs() <= context_size)].reset_index()

    # Add URLs to patches
    patch_urls = get_patch_urls(top_k_nearest.patch_filename)
    top_k_nearest["patch_url"] = top_k_nearest.patch_filename.map(patch_urls)

    if return_tile_size:
        return top_k_nearest, tile_size
    return top_k_nearest


class PatchContext:
    pass


def merge_patch_context(
        patch_filename: str,
        patch_metadata: pd.DataFrame,
        context_size: int = 1,
        bg_color=(255, 255, 255),
) -> tuple[str, PatchContext]:
    """
    Merge patch with its nearest context.
    """
    # Get patch context metadata
    patch_context = get_patch_context(patch_filename, patch_metadata, context_size)

    print(f"Merging context of {patch_filename}...")

    # Sequentially get images and paste them onto the canvas
    for i, row in patch_context.iterrows():
        name, img = load_patch_to_RAM(row.patch_url, row.patch_filename)
        img = img.convert("RGBA")
        x, y = row["x_pos"] + context_size, row["y_pos"] + context_size

        if i == 0:
            # Use first image to determine width/height
            w, h = img.size

            # Create canvas: 3×3 grid
            grid_w, grid_h = (2*context_size+1) * w, (2*context_size+1) * h
            canvas = Image.new("RGBA", (grid_w, grid_h), bg_color)

        # Paste each image at its relative position
        x, y = int(x * w), int(y * h)
        canvas.paste(img, (x, y), img)

    return (patch_filename, canvas.convert("RGB"))


if __name__ == "__main__":

    IMG_NAME = "TCGA-2H-A9GR-01Z-00-DX1_(48665,27374).jpg"

    patch_metadata = pd.read_parquet("metadata/patch_metadata.parquet")

    context = get_patch_context(IMG_NAME, patch_metadata, context_size=2)
    download_patches(context.patch_filename)

    merged_context = merge_patch_context(context, context_size=2)
    merged_context.save("retrieval/output/merged_grid.png")
