# OCR Maths for ML

A small toolkit for converting a PDF of handwritten/printed math notes ("Maths for ML") into clean, readable Markdown — math notation included — using OCR.

Two conversion paths are provided:

| Script | Engine | Notes |
|---|---|---|
| `gemini_ocr_convert.py` | Gemini (`gemini-2.5-flash`) | Cloud OCR, renders PDF pages to images and asks Gemini to transcribe them, math notation preserved |
| `local_ocr_convert.py` / `convert_math_notes.py` | [Pix2Text](https://github.com/breezedeus/Pix2Text) (local, GPU-accelerated) | Fully offline OCR, resumable page-by-page conversion |

Output lands in `Transcribed_Math_Notes.md` (Gemini path) or `output_text/math_notes_converted.md` (local path).

## Usage

**Gemini path** (requires an API key):

```bash
export GEMINI_API_KEY=your_key_here   # never commit this
python gemini_ocr_convert.py
```

**Local path** (no API key, needs a CUDA GPU for reasonable speed):

```powershell
./setup_gpu.ps1        # installs CUDA-enabled PyTorch + Pix2Text
python local_ocr_convert.py
```

`local_ocr_convert.py` checkpoints progress and resumes from the last converted page if interrupted.

## Requirements

- Python 3.x
- `pix2text`, `torch`, `torchvision`, `PyMuPDF` (`fitz`), `Pillow`, `tqdm`
- For the Gemini path: `google-generativeai` and an API key
- For the local path: an NVIDIA GPU (see `setup_gpu.ps1` for CUDA setup)

## Input

Place the source PDF (`Complete_Notes_Maths_For_ML.pdf`) in the project root before running either script.
