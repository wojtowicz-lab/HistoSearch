import requests
import os
import pandas as pd
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
from retrieval.utils import get_patch_urls


def download_patch(
        patch_filename: str,
        patch_url: str,
        output_dir: str = "retrieval/output"
) -> None:
    """Download a patch from Google Drive based on its filename."""
    # Define output path
    outpath = os.path.join(output_dir, patch_filename)

    # Download image
    response = requests.get(patch_url)
    if response.status_code == 200:
        with open(outpath, "wb") as f:
            f.write(response.content)
        print(f"Download complete: {patch_filename}")
    else:
        print("Failed to download:", response.status_code)


def download_patches(
        patch_filenames: list[str],
        output_dir: str = "retrieval/output",
        mapping_filename: list[str] = "retrieval/data/patch_name_to_drive_id_mapping.parquet",
        num_workers: int = os.cpu_count() * 5,
) -> None:
    """Download patches from Google Drive with parallel execution."""
    # Get patch urls
    patch_urls = get_patch_urls(patch_filenames, mapping_filename)

    # Download patches from Drive with parallel execution
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading patches...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_patch, filename, url, output_dir): url for filename, url in zip(patch_filenames, patch_urls)}

        # for future in as_completed(futures):
        #     print(future.result())


class Patch:
    pass


def load_patch_to_RAM(
        patch_url: str,
        patch_filename: str,
        max_retries: int = 3,
        verbose: bool = True,
) -> tuple[str, Patch]:
    """Load a patch form google drive to RAM."""
    for i in range(max_retries):
        try:
            response = requests.get(patch_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            if verbose:
                print(f"✅ Loaded {patch_filename} ({img.format}, {img.size})")
            return (patch_filename, img)
        except Exception as e:
            if verbose: 
                print(f"❌ Failed to load {patch_filename}: {e}")
                print(f"Retry loading {patch_filename}: attempt {i+1}...")
        
    raise Exception(f"Max number of retries ({max_retries}) reached for loading {patch_filename}! Check your internet connection or try in a few minutes.")



def load_patches_to_RAM(
        patch_filenames: list[str],
        mapping_filename: list[str] = "retrieval/data/patch_name_to_drive_id_mapping.parquet",
        num_workers: int = os.cpu_count() * 5,
        verbose: bool = True,
) -> dict[str: Patch]:
    """Load a list of patches from Google Drive to RAM."""
    # Get patch urls
    patch_urls = get_patch_urls(patch_filenames, mapping_filename)

    patches = {}

    if verbose:
        print("Loading patches form Drive...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_patch_to_RAM, url, filename, 3, verbose) for url, filename in patch_urls.items()]

        for future in as_completed(futures):
            filename, img = future.result()
            if img:
                patches[filename] = img
                if verbose:
                    print(f"✅ Loaded {filename} ({img.format}, {img.size})")

    return patches


if __name__ == "__main__":

    # Test values
    PATCH_FILENAMES = [
        "TCGA-2H-A9GK-01Z-00-DX1_(5069,60831).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(5069,59817).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(5069,58803).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(5069,61845).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(5069,62859).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(6083,53734).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(6083,54748).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(6083,55762).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(6083,56776).jpg",
        "TCGA-2H-A9GF-01Z-00-DX1_(6083,57790).jpg",
    ]

    OUTDIR = "retrieval/output"

    # download_patches(PATCH_FILENAMES)
    patches = load_patches_to_RAM(PATCH_FILENAMES)
    print(patches)
