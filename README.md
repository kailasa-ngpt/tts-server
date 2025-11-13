TTS API Server

This repository provides a minimal FastAPI server that wraps a local Text-to-Speech (TTS) model for inference. It is scaffolded so you can drop in logic from the reference notebooks under `ref_notebooks/`.

Features
- FastAPI server with `/health` and `/tts` endpoints
- JSON or multipart input; returns WAV bytes
- Unsloth-backed Sesame CSM (1B) inference
- CORS enabled for browser clients

Quick Start
1) Create a Python environment
   - Windows (PowerShell):
     - `py -m venv .venv`
     - `.venv\\Scripts\\Activate.ps1`
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2) Install dependencies
   - Base server deps:
     - `pip install -r requirements.txt`
   - Install PyTorch for your platform (choose CUDA/CPU build as appropriate):
     - See: https://pytorch.org/get-started/locally/
   - Optional GPU perf deps (match notebooks):
     - `pip install xformers`

3) Run the server
   - `uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload`

4) Call the API
   - JSON (no voice cloning):
     - `curl -X POST "http://localhost:8000/tts" -H "Content-Type: application/json" -d '{"text":"Hello world","language":"en"}' --output out.wav`
   - Multipart (with reference speaker WAV for voice cloning):
     - `curl -X POST "http://localhost:8000/tts" -F "text=Hello from my custom voice" -F "language=en" -F "speaker_wav=@/path/to/voice.wav" --output out.wav`

Environment Variables (optional)
- `TTS_BACKEND`: Which backend to use. Default: `transformers_csm` (CPU-friendly). Use `unsloth_csm` for GPU.
- `DEVICE`: One of `cuda`, `cpu`, or empty to auto-detect.
- `CSM_MODEL_ID`: Model id for CSM. Default: `unsloth/csm-1b`. You may also use `sesame/csm-1b`.
- `CSM_MAX_SEQ_LEN`: Max sequence length for Unsloth load. Default: `2048`.
- `CSM_LOAD_IN_4BIT`: Set `true` to load in 4-bit (if supported).
- `CSM_MAX_NEW_TOKENS`: Max new tokens for generation. Default: `1024`.
- `HF_HOME`: Cache directory for models/datasets (replaces deprecated `TRANSFORMERS_CACHE`). Example: `/cache/huggingface`.

Integrating Your Notebook Logic
- Replace or extend the implementation in `server/inference.py` with the core steps from your notebook:
  - Model load in `load_model()`
  - Synthesis in `synthesize(...)`
- Keep the function signatures intact so the API layer does not need changes.
- If your notebook uses different inputs/outputs, adjust the Pydantic models in `server/models.py` accordingly.

Notes
- Torch and model downloads are large. The first run may take time.
- If you prefer a different TTS library (e.g., Piper, Bark, Edge-TTS, ElevenLabs), add a new backend class in `server/inference.py` and switch `TTS_BACKEND`.

Sesame/CSM (Unsloth) Notes
- Uses Unsloth `FastModel.from_pretrained(..., auto_model=CsmForConditionalGeneration)` for faster, memory‑efficient loading.
- Prompt format includes audio start tokens; the server composes `<|begin_of_text|>your text<|end_of_text|><|AUDIO|>` by default.
- Pass a short reference clip via `speaker_wav` (multipart) for voice conditioning.
- Default sampling rate is 24kHz.
 - requirements.txt includes extra packages often used by the notebooks (datasets, accelerate, peft, trl). They are not strictly required for server inference but included per your request.
