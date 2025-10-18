# Persian TTS Web Service

## Requirements
- A machine with a GPU and NVIDIA drivers (recommended). The notebook originally used PyTorch 1.13.1 with CUDA 11.7.
- Docker + NVIDIA Container Toolkit if using Docker GPU container.
- Or Python 3.8+ installed locally.

## Quick local run (without docker)
1. Create a virtual environment and activate it:
   ```bash
   python3.8 -m venv venv
   source venv/bin/activate
   ```
2. Install PyTorch with CUDA (if you have CUDA 11.7):
   ```bash
   python3.8 -m pip install --upgrade pip
   python3.8 -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 "torch==1.13.1+cu117" "torchaudio==0.13.1"
   ```
   
3. Run:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Open `http://localhost:8000` in your browser.

## Run with Docker (GPU)
1. Build:
   ```bash
   docker build -t persian-tts:latest .
   ```
2. Run:
   ```bash
   docker run --gpus all -p 8000:8000 persian-tts:latest
   ```
3. Visit `http://<host-ip>:8000`.

## Notes & Caveats
- The model `facebook/mms-tts-fas` is relatively large and benefits from GPU memory (8GB+ recommended). 
- If deploying publicly, consider:
  - Using a GPU VM (e.g., cloud VM with GPU) and running Docker with `--gpus`.
  - Using a managed inference option (Hugging Face Inference endpoints / Spaces with GPU support) for production traffic.
- For production, add rate-limiting, authentication, input length checks, and background cleanup.
