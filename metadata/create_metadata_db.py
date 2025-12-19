import sqlite3
import pandas as pd


if __name__ == "__main__":
    # Load patch and patient metadata
    patch_metadata_file = "metadata/raw_data/uni2h_augmented_IndexIVFPQ_patch_metadata.parquet"
    patient_metadata_file = "metadata/raw_data/liu.csv"

    # Read the dataframes
    patch_metadata_df = pd.read_parquet(patch_metadata_file)
    patient_metadata_df = pd.read_csv(patient_metadata_file)

    print(patch_metadata_df)
    print(patient_metadata_df)

    # # Create patient metadata db
    # patient_metadata_df.columns = [col.replace(" ", "_") for col in patient_metadata_df.columns]
    # conn = sqlite3.connect("metadata/metadata.db")
    # patient_metadata_df.to_sql("patients", conn, if_exists="replace", index=False)

    # Create patch metadata DataFrame
    metadata_df = patch_metadata_df.copy()
    metadata_df.columns = ["faiss_index", "slide_id", "embedding", "rotation"]
    metadata_df[['patch_coord_x','patch_coord_y']] = metadata_df['embedding'].str.extract(r'\((\d+),(\d+)\)').astype(int)
    metadata_df["slide"] = metadata_df['embedding'].str.extract(r'(.+)_')
    metadata_df["patch_filename"] = metadata_df["embedding"] + ".jpg"
    metadata_df.drop(columns="embedding", inplace=True)
    metadata_df["patient_id"] = metadata_df["slide"].str[:12]
    # metadata_df.drop(columns="MSI_status", inplace=True)

    metadata_df.to_parquet("metadata/patch_metadata.parquet")

    # Add signatures to patient metadata
    signature_df = pd.read_csv("metadata/raw_data/Assignment_Solution_Activities.txt", sep="\t", index_col=0)
    signature_bool_df = ((signature_df > 0) * 1)

    merged_patient_metadata_df = pd.merge(patient_metadata_df, signature_bool_df, left_on="TCGA Participant Barcode", right_index=True)
    merged_patient_metadata_df.to_parquet("metadata/patient_metadata.parquet")
