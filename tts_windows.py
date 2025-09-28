import os
import logging
import platform
from typing import Optional
import numpy as np
from scipy import signal
from scipy.io import wavfile
import onnxruntime as ort

logger = logging.getLogger(__name__)

def _init_sapi():
    """Initialize Windows SAPI if available"""
    try:
        import win32com.client
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        return voice, stream
    except Exception as e:
        logger.warning(f"Failed to initialize SAPI: {e}")
        return None, None

def _ensure_16khz_mono_16bit(audio: np.ndarray, sr: int) -> np.ndarray:
    """Ensure audio is 16kHz mono 16-bit"""
    if sr != 16000:
        # Resample to 16kHz
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Normalize and convert to 16-bit
    audio = np.clip(audio, -1, 1)
    return (audio * 32767).astype(np.int16)

def _griffin_lim(mel_spectrogram: np.ndarray, n_iter: int = 32) -> np.ndarray:
    """Basic Griffin-Lim algorithm for mel spectrogram inversion"""
    n_fft = (mel_spectrogram.shape[0] - 1) * 2
    angles = np.exp(2j * np.pi * np.random.rand(*mel_spectrogram.shape))
    
    for _ in range(n_iter):
        S = mel_spectrogram * angles
        y = signal.istft(S, nperseg=n_fft)[1]
        angles = np.exp(1j * np.angle(signal.stft(y, nperseg=n_fft)[2]))
    
    return signal.istft(mel_spectrogram * angles, nperseg=n_fft)[1]

def generate_tts(
    text: str,
    out_wav: str,
    voice: Optional[str] = None,
    sample_rate: int = 16000,
    model_dir: Optional[str] = None,
    prefer_sapi: bool = True,
    ssml: bool = False
) -> str:
    """Generate TTS audio and save as WAV file
    
    Args:
        text: Input text to synthesize
        out_wav: Output WAV file path 
        voice: Voice name for SAPI or model identifier
        sample_rate: Target sample rate (will be converted to 16kHz)
        model_dir: Directory containing ONNX models if not using SAPI
        prefer_sapi: Whether to prefer Windows SAPI over ONNX
        ssml: Whether input text is SSML
        
    Returns:
        Absolute path to generated WAV file
        
    Raises:
        RuntimeError: If TTS generation fails
        FileNotFoundError: If ONNX models not found
        ValueError: If invalid parameters provided
    """
    if not text:
        raise ValueError("Empty text provided")
    
    out_wav = os.path.abspath(out_wav)
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    
    # Try SAPI if preferred and on Windows
    if prefer_sapi and platform.system() == "Windows":
        voice_obj, stream = _init_sapi()
        if voice_obj and stream:
            try:
                stream.Open(out_wav, 0, False)
                voice_obj.AudioOutputStream = stream
                
                if voice:
                    for v in voice_obj.GetVoices():
                        if voice in v.GetDescription():
                            voice_obj.Voice = v
                            break
                
                # Speak text
                flags = 8 if ssml else 0  # 8 = SAPI XML flag
                voice_obj.Speak(text, flags)
                stream.Close()
                return out_wav
            except Exception as e:
                logger.error(f"SAPI synthesis failed: {e}")
                # Fall through to ONNX
    
    # Fall back to ONNX if SAPI failed or not preferred
    if not model_dir:
        raise RuntimeError(
            "ONNX model directory required when SAPI unavailable or not preferred. "
            "Place text2mel.onnx and vocoder.onnx in model directory."
        )
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    # Initialize ONNX Runtime session
    providers = ort.get_available_providers()
    if 'QNNExecutionProvider' in providers:
        logger.info("Using Qualcomm Neural Network provider")
        provider = ['QNNExecutionProvider']
    else:
        provider = ['CPUExecutionProvider']
        
    # Load text2mel model
    text2mel_path = os.path.join(model_dir, "text2mel.onnx")
    vocoder_path = os.path.join(model_dir, "vocoder.onnx") 
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    
    if not os.path.exists(text2mel_path):
        raise FileNotFoundError(
            f"text2mel.onnx not found in {model_dir}. "
            "Place your text-to-mel spectrogram ONNX model here."
        )
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"tokenizer.json not found in {model_dir}. "
            "Place your tokenizer configuration file here."
        )
    
    try:
        # Load and run text2mel model
        text2mel = ort.InferenceSession(text2mel_path, providers=provider)
        
        # NOTE: Implement tokenization based on your tokenizer.json
        # This is a placeholder - replace with actual tokenization
        token_ids = np.array([ord(c) for c in text], dtype=np.int64)[None, :]
        
        # Generate mel spectrogram
        mel = text2mel.run(None, {"input_ids": token_ids})[0]
        
        # Use vocoder if available, otherwise Griffin-Lim
        if os.path.exists(vocoder_path):
            vocoder = ort.InferenceSession(vocoder_path, providers=provider)
            audio = vocoder.run(None, {"mel": mel})[0]
        else:
            logger.warning("Vocoder not found, using Griffin-Lim algorithm")
            audio = _griffin_lim(mel[0].T)
        
        # Ensure audio format and save
        audio = _ensure_16khz_mono_16bit(audio, sample_rate)
        wavfile.write(out_wav, 16000, audio)
        return out_wav
        
    except Exception as e:
        raise RuntimeError(f"ONNX inference failed: {e}")

def play_local(out_wav: str) -> None:
    """Play WAV file using Windows APIs
    
    Args:
        out_wav: Path to WAV file to play
        
    Note:
        This function is disabled by default in CI/test environments
        by checking for CI environment variables
    """
    if os.environ.get("CI") or os.environ.get("TEST_ENV"):
        logger.info("Playback disabled in CI/test environment")
        return
        
    if platform.system() != "Windows":
        logger.warning("Local playback only supported on Windows")
        return
        
    try:
        import winsound
        winsound.PlaySound(out_wav, winsound.SND_FILENAME)
    except Exception as e:
        logger.error(f"Failed to play audio: {e}")