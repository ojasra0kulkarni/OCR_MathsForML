import os
from pix2text import Pix2Text
from tqdm import tqdm
import torch

# Configuration
PDF_PATH = "Complete_Notes_Maths_For_ML.pdf"
OUTPUT_DIR = "output_text"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "math_notes_converted.md")

def check_gpu():
    print("Checking GPU Status...")
    if torch.cuda.is_available():
        print(f"CUDA is available! Using: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("CUDA NOT available. Using CPU (this will be slower).")
        return "cpu"

def main():
    # 1. Initialize result directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 2. Check for GPU
    device = check_gpu()

    # 3. Initialize Pix2Text
    print("Initializing Pix2Text model (this may take a moment to load and download models if first run)...")
    p2t = Pix2Text(device=device)

    # 4. Recognize PDF
    # Note: recognize_pdf handles full documents. For 144 pages, we process and save.
    print(f"Starting OCR for {PDF_PATH} (144 pages)...")
    
    try:
        # recognize_pdf automatically converts pages to images and processes them.
        # You can specify page_numbers=[0, 1, 2] if you want to test specific pages first.
        doc = p2t.recognize_pdf(PDF_PATH)
        
        # 5. Export to Markdown
        print(f"Exporting results to {OUTPUT_DIR}...")
        doc.to_markdown(OUTPUT_DIR)
        
        # doc.to_markdown creates a directory with .md and images.
        # We can also handle the final renaming if needed.
        print("\n" + "="*50)
        print(f"SUCCESS! Conversion complete.")
        print(f"Your converted notes are in: {OUTPUT_DIR}")
        print("="*50)

    except Exception as e:
        print(f"Error during OCR: {e}")

if __name__ == "__main__":
    main()
