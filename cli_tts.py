import argparse
import logging
import os
from tts_windows import generate_tts, play_local

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Windows TTS CLI")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", required=True, help="Output WAV file")
    parser.add_argument("--voice", help="Voice name (SAPI) or identifier")
    parser.add_argument("--model-dir", help="Directory containing ONNX models")
    parser.add_argument("--no-sapi", action="store_true", help="Disable SAPI fallback")
    parser.add_argument("--ssml", action="store_true", help="Input is SSML")
    parser.add_argument("--play", action="store_true", help="Play audio after generation")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    
    args = parser.parse_args()
    
    try:
        out_path = generate_tts(
            text=args.text,
            out_wav=args.out,
            voice=args.voice,
            model_dir=args.model_dir,
            prefer_sapi=not args.no_sapi,
            ssml=args.ssml,
            sample_rate=args.sample_rate
        )
        
        logger.info(f"Audio saved to: {os.path.abspath(out_path)}")
        
        if args.play:
            play_local(out_path)
            
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()