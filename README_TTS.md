# Windows-First Text-to-Speech Module

A high-performance, offline Text-to-Speech (TTS) module designed for Windows, with fallback to ONNX models.

## Quick Start

```python
from tts_windows import generate_tts

# Using Windows SAPI (fastest)
out = generate_tts(
    "Room 203 is on the second floor.",
    r"C:\projects\athena\out\room203.wav",
    voice="Microsoft Zira Desktop"  # Optional: specify SAPI voice
)

# Using ONNX models (offline, custom voices)
out = generate_tts(
    "Room 203 is on the second floor.",
    "out.wav",
    model_dir="models",  # Directory containing ONNX models
    prefer_sapi=False    # Force ONNX pipeline
)
```

## Features

- Windows SAPI support (primary, fastest option)
- ONNX model support with Qualcomm Neural Network acceleration
- 16 kHz, mono, 16-bit PCM WAV output
- SSML support with SAPI
- CLI interface

## Installation

```bash
pip install -r requirements.txt
```

## ONNX Model Setup

Place your ONNX models in a directory with this structure:

```
models/
  ├── text2mel.onnx      # Text to mel-spectrogram model
  ├── vocoder.onnx       # Optional: mel to waveform model
  └── tokenizer.json     # Tokenizer configuration
```

If `vocoder.onnx` is not provided, the system will use Griffin-Lim algorithm for mel-spectrogram inversion.

## CLI Usage

```powershell
# Using SAPI
python cli_tts.py --text "Hello, world!" --out test.wav --voice "Microsoft Zira Desktop"

# Using ONNX
python cli_tts.py --text "Hello, world!" --out test.wav --model-dir models --no-sapi

# Play audio after generation
python cli_tts.py --text "Hello, world!" --out test.wav --play

# Use SSML
python cli_tts.py --text "<speak>Hello <emphasis>world</emphasis>!</speak>" --ssml --out test.wav
```

## Packaging with PyInstaller

```powershell
# Basic
pyinstaller --onefile cli_tts.py

# Include ONNX models
pyinstaller --onefile --add-data "models;models" cli_tts.py
```

Add to your spec file:

```python
a.datas += Tree('./models', prefix='models')
```

## Performance Notes

1. SAPI Performance:
   - Latency: 200-500ms for typical phrases
   - No GPU/special hardware required
   - Best choice for Windows deployments

2. ONNX Performance:
   - CPU: 0.5-2s depending on text length
   - With Qualcomm AI Engine: ~200-400ms
   - Memory usage depends on model size

3. Optimization Tips:
   - Use quantized ONNX models (INT8/FP16)
   - Enable Qualcomm AI Engine when available
   - For low latency, prefer SAPI on Windows

## Error Handling

Common errors and solutions:

1. "SAPI not available":
   - Ensure running on Windows
   - Check Windows Text-to-Speech settings

2. "ONNX model not found":
   - Verify model directory structure
   - Check model filenames match expected

3. "Memory allocation failed":
   - Try quantized models
   - Reduce batch size
   - Check available system memory

## Testing

```bash
pytest tests/test_tts_basic.py
```

## License

MIT