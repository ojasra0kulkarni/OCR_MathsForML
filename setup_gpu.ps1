# OCR Setup Script for GPU Support
# This script will attempt to install the CUDA-enabled version of PyTorch and Pix2Text.

Write-Host "Checking for CUDA compatibility..." -ForegroundColor Cyan
nvidia-smi

Write-Host "`nStep 1: Uninstalling existing PyTorch (to avoid conflicts)..." -ForegroundColor Yellow
pip uninstall torch torchvision torchaudio -y

Write-Host "`nStep 2: Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
# Using --pre or nightly if stable is not available for Python 3.13
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "`nStep 3: Installing Pix2Text and OCR dependencies..." -ForegroundColor Yellow
pip install pix2text[multilingual]

Write-Host "`nStep 4: Verifying Installation..." -ForegroundColor Yellow
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSuccess! Your environment is ready for GPU-accelerated OCR." -ForegroundColor Green
} else {
    Write-Host "`nWarning: CUDA might still be disabled. Check if your NVIDIA drivers are up to date." -ForegroundColor Red
}
