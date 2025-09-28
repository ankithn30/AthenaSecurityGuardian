import os
import tempfile
import pytest
import numpy as np
from scipy.io import wavfile
from unittest.mock import patch, MagicMock

# Add parent directory to path to import tts_windows
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_windows import generate_tts, _griffin_lim

@pytest.fixture
def temp_wav():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def mock_model_dir(tmp_path):
    """Create mock ONNX model files"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Create dummy ONNX files
    (model_dir / "text2mel.onnx").write_bytes(b"dummy")
    (model_dir / "vocoder.onnx").write_bytes(b"dummy")
    (model_dir / "tokenizer.json").write_bytes(b"{}")
    
    return str(model_dir)

def test_griffin_lim():
    """Test Griffin-Lim algorithm"""
    mel = np.random.rand(80, 100)  # Dummy mel spectrogram
    audio = _griffin_lim(mel, n_iter=2)
    assert isinstance(audio, np.ndarray)
    assert len(audio.shape) == 1

@patch('onnxruntime.InferenceSession')
def test_generate_tts_onnx(mock_session, temp_wav, mock_model_dir):
    """Test TTS generation with ONNX pipeline"""
    # Mock ONNX inference
    mock_session.return_value.run.return_value = [np.zeros((1, 80, 100))]
    
    # Generate TTS with SAPI disabled
    out_path = generate_tts(
        "Test text",
        temp_wav,
        model_dir=mock_model_dir,
        prefer_sapi=False
    )
    
    # Verify output
    assert os.path.exists(out_path)
    sr, audio = wavfile.read(out_path)
    assert sr == 16000
    assert audio.dtype == np.int16
    assert len(audio) > 0

def test_generate_tts_errors():
    """Test error handling"""
    with pytest.raises(ValueError):
        generate_tts("", "test.wav")
        
    with pytest.raises(RuntimeError):
        generate_tts("test", "test.wav", prefer_sapi=False)
        
    with pytest.raises(FileNotFoundError):
        generate_tts("test", "test.wav", model_dir="/nonexistent", prefer_sapi=False)

if __name__ == "__main__":
    pytest.main([__file__])