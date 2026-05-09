import pandas as pd


def get_filename_to_id_mapping(
        mapping_filename: str = "retrieval/data/patch_name_to_drive_id_mapping.parquet"
) -> pd.Series:
    """Load patch name to Google Drive file id mapping as pd.Series."""
    mapping = pd.read_parquet(mapping_filename).set_index("filename")
    return mapping["file_id"]


def get_patch_urls(
        patch_filenames: list[str],
        mapping_filename: str = "retrieval/data/patch_name_to_drive_id_mapping.parquet"
) -> list[str]:
    """Fucntion returning an URL to download a patch from the database based on patch_filename."""
    # Get filename to Google Drive ID mapping
    mapping = get_filename_to_id_mapping(mapping_filename)

    # Template for downloading a photo from Google Drive
    DOWNLOAD_URL_TEMPLATE = "https://drive.google.com/uc?export=download&id={file_id}"
    
    # Get Google Drive file ids
    file_ids = mapping.loc[patch_filenames].to_list()

    return {filename: DOWNLOAD_URL_TEMPLATE.format(file_id=file_id) for filename, file_id in zip(patch_filenames, file_ids)}


def determine_tile_size(context_df: pd.DataFrame, verbose: bool = False) -> int:
    """Determine tile size based on the most common distance between neighboring patches in the metadata."""
    # Get the most common non-zero distance
    x_tile_size = context_df.loc[context_df["x_diff"] > 0, "x_diff"].mode()[0]
    y_tile_size = context_df.loc[context_df["y_diff"] > 0, "y_diff"].mode()[0]

    if verbose:
        print(f"Determined tile size: x={x_tile_size}, y={y_tile_size}")

        if x_tile_size != y_tile_size:
            print(f"Warning: Determined different tile sizes for x and y directions: {x_tile_size} vs {y_tile_size}. Using the smaller one.")
    
    return int(min(x_tile_size, y_tile_size))