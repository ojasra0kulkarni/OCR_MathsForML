import os
import time
import fitz  # PyMuPDF
import google.generativeai as genai
from tqdm import tqdm
from PIL import Image
import io

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
PDF_PATH = "Complete_Notes_Maths_For_ML.pdf"
MODEL_NAME = "gemini-2.5-flash"
OUTPUT_FILE = "Transcribed_Math_Notes.md"

# Safety settings to avoid false positives on math notes
SAFETY_SETTINGS = {
    "HATE": "BLOCK_NONE",
    "HARASSMENT": "BLOCK_NONE",
    "SEXUAL": "BLOCK_NONE",
    "DANGEROUS": "BLOCK_NONE",
}

PROMPT = """
Transcribe the handwriting in this image into a clean Markdown document.
Rules:
1. Handle all text as accurately as possible.
2. If there are mathematical formulas, transcribe them into precise LaTeX.
3. Use $ ... $ for inline math and $$ ... $$ for displayed block equations.
4. Maintain the structure and flow of the original notes.
5. Do not add any conversational text or explanations; just provide the transcription.
"""

# Manual override for pages that were skipped
MISSING_PAGES = [20, 21, 22, 23, 24, 25, 27]
RESUME_PAGE = 28 # After backfilling, start from here

def transcribe_page(model, doc, page_idx):
    """Try to transcribe a single page with retries for rate limits."""
    retry_count = 0
    while True:
        try:
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            response = model.generate_content([PROMPT, img])
            text = response.text
            
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"## Page {page_idx + 1}\n\n{text}\n\n---\n\n")
            
            return True

        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e) or "Quota" in str(e):
                tqdm.write(f"\nRate limit hit on page {page_idx + 1}. Waiting 60s...")
                time.sleep(60)
            else:
                tqdm.write(f"\nError on page {page_idx + 1}: {e}")
                retry_count += 1
                if retry_count >= 3:
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"## Page {page_idx + 1}\n\n[OCR ERROR: {e}]\n\n---\n\n")
                    return False
                time.sleep(10)

def sort_markdown(filepath):
    import re
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parts = re.split(r'(?m)^## Page ', content)
    header = parts[0]
    pages = parts[1:]
    
    parsed_pages = []
    for p in pages:
        match = re.search(r'^(\d+)', p)
        if match:
            num = int(match.group(1))
            parsed_pages.append((num, p))
    
    parsed_pages.sort(key=lambda x: x[0])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header)
        for num, p in parsed_pages:
            f.write(f"## Page {p}")
            if not p.endswith('\n'):
                f.write('\n')

def main():
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME, safety_settings=SAFETY_SETTINGS)
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)

    print(f"Starting OCR. Target: {total_pages} pages.")

    # 1. Backfill missing pages first
    if MISSING_PAGES:
        print(f"Backfilling missing pages: {MISSING_PAGES}")
        for p in tqdm(MISSING_PAGES, desc="Backfilling"):
            transcribe_page(model, doc, p - 1) 
            time.sleep(10) # Heavy delay for stability

    # 2. Resume normal process
    print(f"Resuming from page {RESUME_PAGE} to {total_pages}")
    for p_idx in tqdm(range(RESUME_PAGE - 1, total_pages), desc="Resuming OCR"):
        transcribe_page(model, doc, p_idx)
        time.sleep(10) # Heavy delay for stability

    doc.close()
    
    print("Sorting markdown file...")
    sort_markdown(OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    main()
