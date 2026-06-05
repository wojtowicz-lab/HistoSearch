import gradio as gr
import numpy as np
import pandas as pd
import os
import shutil
from PIL import Image

from models.utils import load_UNI2h
from embeddings.embed import get_UNI2h_patch_embedding
from faiss_search.search import get_most_similar_patches
from retrieval.context import merge_patch_context
from metadata.utils import get_metadata, create_metadata_filter
from concurrent.futures import ThreadPoolExecutor, as_completed


# Global variables
INDEX_FILEPATH = "faiss_search/faiss_indices/uni2h_IndexIVFPQ_augmented.faiss"
OUTPUT_DIR = "retrieval/output"

# Load model once at startup
print("🚀 Loading UNI2-h model at startup...")
MODEL, TRANSFORM = load_UNI2h()
print("✅ Model loaded and ready!")


def process_image_query(
    img_path: str,
    n_patients: int,
    n_patches: int,
    project_filter: list[str],
    organ_filter: list[str],
    stage_filter: list[str],
    gender_filter: list[str],
    vital_status_filter: list[str],
    anatomic_region_filter: list[str],
    msi_status_filter: list[str],
    mutational_signature_filter: list[str],
    selected_display_columns: list[str],
    selected_signature_columns: list[str],
    context_size: int = 1,
    max_workers: int = 8,
):
    """
    Process uploaded image and return similar patches.
    
    Args:
        image: Uploaded PIL Image
        n_patients: Number of similar patients to retrieve
        n_patches: Number of patches per patient
        organ_filter: List of organ types to filter by
        gender_filter: List of genders to filter by
    
    Returns:
        gallery_images: List of image paths for the gallery
        results_df: DataFrame with metadata
        status_text: Status message
    """
    try:
        # Reset output directory
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)
        
        # Build filters dictionary
        filters = create_metadata_filter(
            project_filter=project_filter,
            organ_filter=organ_filter,
            stage_filter=stage_filter,
            gender_filter=gender_filter,
            vital_status_filter=vital_status_filter,
            anatomic_region_filter=anatomic_region_filter,
            msi_status_filter=msi_status_filter,
            mutational_signature_filter=mutational_signature_filter,
        )
        
        # Get filtered patient metadata
        status = "📊 Loading metadata...\n"
        patient_metadata, patch_metadata = get_metadata(filters)
        status += f"✓ Found {len(patient_metadata)} patients matching criteria\n"
        
        # Get query patch embedding
        status += "🧬 Creating image embedding...\n"
        query_vec = get_UNI2h_patch_embedding(img_path, model=MODEL, transform=TRANSFORM)
        status += "✓ Embedding created\n"
        
        # Get most similar patches
        status += "🔍 Searching for similar patches...\n"
        search_results = get_most_similar_patches(
            query_vec=query_vec,
            patient_metadata=patient_metadata,
            patch_metadata=patch_metadata,
            n_patients=n_patients,
            n_patches=n_patches,
            index_filepath=INDEX_FILEPATH,
        )
        status += f"✓ Found {len(search_results)} similar patches\n"

        # Delete augmented tiles from metadata to avoid repetitions
        unaugmented_patch_metadata = patch_metadata.drop_duplicates("patch_filename")
        
        # Download patches with their context
        status += "⬇️ Downloading and merging matched patches and their context...\n"
        processed_patches = set()  # Track processed patches to avoid duplicates
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for filename in search_results.patch_filename:
                if filename not in processed_patches:
                    processed_patches.add(filename)
                    futures.append(executor.submit(merge_patch_context, filename, unaugmented_patch_metadata, context_size))

            for future in as_completed(futures):
                filename, img = future.result()
                name, ext = os.path.splitext(filename)
                img.save(os.path.join(OUTPUT_DIR, f"{name}_context{ext}"))
        status += "✓ Download complete\n"            
        
        # Prepare gallery images
        gallery_images = []
        for filename in search_results["patch_filename"]:
            name, ext = os.path.splitext(filename)
            img_path = os.path.join(OUTPUT_DIR, f"{name}_context{ext}")
            if os.path.exists(img_path):
                gallery_images.append(img_path)
        
        # Prepare results dataframe for display
        display_columns = ['patch_filename', 'score', 'patient_id']

        # Append user-selected metadata columns
        if selected_display_columns:
            for col in selected_display_columns:
                if col not in display_columns:
                    display_columns.append(col)

        # Append user-selected mutational signatures to display
        if selected_signature_columns:
            for sig in selected_signature_columns:
                if sig not in display_columns:
                    display_columns.append(sig)
        available_columns = [col for col in display_columns if col in search_results.columns]
        results_display = search_results[available_columns].copy()
        results_display['score'] = results_display['score'].round(4)
        
        status += f"\n✅ Successfully retrieved {len(gallery_images)} images!"
        
        return gallery_images, results_display, status
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return [], pd.DataFrame(), error_msg


# Available filter options
PROJECT_OPTIONS = ['ESCA', 'COAD', 'STAD', 'READ']
ORGAN_OPTIONS = ['Esophageal', 'COAD', 'Gastric', 'READ']
STAGE_OPTIONS = ['I', 'II', 'III', 'IV']
# COUNTRY_OPTIONS = ['Netherlands', 'United_States', 'Brazil', 'Germany', 'Russia', 'Ukraine',
#                    'Vietnam', 'Poland', 'Israel', 'Canada', 'Korea_South', 'Australia', 'Moldova', 'United_Kingdom']
GENDER_OPTIONS = ["MALE", "FEMALE"]
VITAL_STATUS_OPTIONS = ['Dead', 'Alive']
ANATOMIC_REGION_OPTIONS = ['Esophagus', 'Ascending_Colon', 'Gastric_Fundus_and_Body',
 'Descending_Colon', 'Transverse_Colon', 'Rectum',
 'Gastric_Antrum_and_Pylorus', 'Gastroesophageal_Junction',
 'Proximal_Stomach']
MSI_STATUS_OPTIONS = ['MSI-L', 'MSS', 'MSI-H']
MUTATIONAL_SIGNATURE_OPTIONS = ["SBS1","SBS2","SBS3","SBS5","SBS6","SBS10a","SBS10b","SBS13","SBS15",
                                "SBS17a","SBS17b","SBS18","SBS20","SBS21","SBS28","SBS29","SBS40a"]

# Columns available for display (keys from create_metadata_filter)
META_DISPLAY_OPTIONS = [
    "TCGA Project Code",
    "Organ",
    "Stage",
    "Gender",
    "Vital status",
    "Anatomic Region",
    "MSI Status",
]


# Create Gradio interface
with gr.Blocks(title="Histopathology Image Retrieval", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Histopathology Image Retrieval System
        Upload a histopathology image patch (JPG) and find similar patches from the TCGA database.
        The system uses UNI2-h embeddings and FAISS for efficient similarity search.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image upload
            gr.Markdown("### 📤 Upload Query Image")
            image_input = gr.Image(
                type="filepath",
                label="Upload Histopathology Patch (JPG)",
                height=300
            )
            
            # Search parameters
            gr.Markdown("### ⚙️ Search Parameters")
            n_patients = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Patients",
                info="Top N similar patients to retrieve"
            )
            
            n_patches = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Patches per Patient",
                info="Number of patches per patient"
            )

            context_size = gr.Slider(
                minimum=0,
                maximum=3,
                value=1,
                step=1,
                label="Context Size",
                info="Number of surrounding patch layers to display"
            )

            max_workers = gr.Slider(
                minimum=0,
                maximum=min(32, (os.cpu_count() or 1) + 4),
                value=max(8, (os.cpu_count() or 1) + 4),
                step=1,
                label="Max Workers",
                info="Number of threads used for parallel processing"
            )

            # Filter choice
            gr.Markdown("### 🔍 Filters (Optional)")
            project_filter = gr.CheckboxGroup(
                choices=PROJECT_OPTIONS,
                label="TCGA Project Code",
                info="Filter by TCGA Project"
            )

            organ_filter = gr.CheckboxGroup(
                choices=ORGAN_OPTIONS,
                label="Organ Type",
                info="Filter by organ type"
            )

            stage_filter = gr.CheckboxGroup(
                choices=STAGE_OPTIONS,
                label="Pathologic stage",
                info="Filter by pathologic stage"
            )
            
            gender_filter = gr.CheckboxGroup(
                choices=GENDER_OPTIONS,
                label="Gender",
                info="Filter by patient gender"
            )

            vital_status_filter = gr.CheckboxGroup(
                choices=VITAL_STATUS_OPTIONS,
                label="Vital status",
                info="Filter by vital status"
            )

            anatomic_region_filter = gr.CheckboxGroup(
                choices=ANATOMIC_REGION_OPTIONS,
                label="Anatomic Region",
                info="Filter by specific anatomi region"
            )

            msi_status_filter = gr.CheckboxGroup(
                choices=MSI_STATUS_OPTIONS,
                label="MSI Status",
                info="Filter by patient MSI Status"
            )

            mutational_signature_filter = gr.CheckboxGroup(
                choices=MUTATIONAL_SIGNATURE_OPTIONS,
                label="Mutational Signatures",
                info="Filter by Mutational Signature Activity (etiology)"
            )
            
            display_columns_checkbox = gr.CheckboxGroup(
                choices=META_DISPLAY_OPTIONS,
                label="Columns to include in results",
                info="Select which metadata columns to show in the results table"
            )

            signatures_display_checkbox = gr.CheckboxGroup(
                choices=MUTATIONAL_SIGNATURE_OPTIONS,
                label="Mutational Signatures to include in results",
                info="Select which mutational signature columns to show in the results table"
            )
            
            # Search button
            search_btn = gr.Button("🔍 Search Similar Patches", variant="primary", size="lg")
            
            status_output = gr.Textbox(
                label="Status",
                lines=10,
                max_lines=15,
                interactive=False
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Retrieved Similar Patches")
            gallery_output = gr.Gallery(
                label="Similar Patches",
                show_label=True,
                elem_id="gallery",
                columns=3,
                rows=3,
                height="auto",
                object_fit="contain"
            )
            
            gr.Markdown("### 📋 Metadata")
            dataframe_output = gr.Dataframe(
                label="Search Results",
                wrap=True
            )
    
    # Examples
    gr.Markdown("### 💡 Example")
    gr.Examples(
        examples=[
            [
                "embeddings/TCGA-2H-A9GK-01Z-00-DX1_(5069,60831).jpg",
                5,
                5,
                [],
                ["Esophageal", "COAD"],
                [],
                ["MALE"],
                [],
                [],
                [],
                [],
                [],
                [],
                1,
                8,
            ]
        ],
        inputs=[
            image_input, n_patients, n_patches,
            project_filter,
            organ_filter,
            stage_filter,
            gender_filter,
            vital_status_filter,
            anatomic_region_filter,
            msi_status_filter,
            mutational_signature_filter,
            display_columns_checkbox,
            signatures_display_checkbox,
            context_size,
            max_workers,
        ],
        label="Try this example"
    )
    
    # Connect the button
    search_btn.click(
        fn=process_image_query,
        inputs=[
            image_input, n_patients, n_patches,
            project_filter,
            organ_filter,
            stage_filter,
            gender_filter,
            vital_status_filter,
            anatomic_region_filter,
            msi_status_filter,
            mutational_signature_filter,
            display_columns_checkbox,
            signatures_display_checkbox,
            context_size,
            max_workers,
        ],
        outputs=[gallery_output, dataframe_output, status_output]
    )
    
    gr.Markdown(
        """
        ---
        ### 📖 About
        This system performs content-based image retrieval on histopathology patches:
        1. **Upload** a query image patch
        2. **Extract** features using UNI2-h model (1536-dim embeddings)
        3. **Search** FAISS index for similar patches
        4. **Retrieve** and display results with metadata
        
        Data source: TCGA (The Cancer Genome Atlas)
        """
    )


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Launch the app
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

