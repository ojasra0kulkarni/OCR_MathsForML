import os
import fitz  # PyMuPDF
from PIL import Image
import io
import re
from tqdm import tqdm
from pix2text import Pix2Text, merge_line_texts

PDF_PATH = "Complete_Notes_Maths_For_ML.pdf"
OUTPUT_FILE = "Transcribed_Math_Notes.md"

def get_resume_page():
    if not os.path.exists(OUTPUT_FILE):
        return 1
    
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all page numbers written as ## Page X
    matches = re.findall(r'^## Page (\d+)', content, re.MULTILINE)
    if matches:
        max_page = max(int(m) for m in matches)
        return max_page + 1
    return 1

def main():
    print("Loading Pix2Text offline model (may take a moment to initialize on GPU)...")
    
    # Initialize Pix2Text with default config (auto-detects CUDA)
    p2t = Pix2Text.from_config()
    
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    
    start_page = get_resume_page()
    
    print(f"Total pages: {total_pages}. Resuming locally from Page {start_page}...")
    
    if start_page > total_pages:
        print("All pages have already been transcribed!")
        return

    for p_idx in tqdm(range(start_page - 1, total_pages), desc="Offline OCR"):
        page = doc[p_idx]
        
        # Pix2Text benefits from higher resolution scale
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        try:
            # Use general mixed model to extract LaTeX block + math
            if hasattr(p2t, 'recognize_text_formula'):
                text = p2t.recognize_text_formula(img, return_text=True)
            else:
                outs = p2t.recognize(img)
                text = merge_line_texts(outs, auto_line_break=True)
                
            if isinstance(text, dict):
                # Fallback if return_text=True doesn't return string directly
                text = str(text)

        except Exception as e:
            text = f"[LOCAL OCR ERROR: {e}]"
            
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"## Page {p_idx + 1}\n\n{text}\n\n---\n\n")

    doc.close()
    print("Finished extracting all pages offline!")

if __name__ == "__main__":
    main()
