import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Audio Settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    CHANNELS = 1
    AUDIO_FORMAT = "int16"
    
    # Model Settings
    WHISPER_MODEL = "openai/whisper-small"
    BERT_MODEL = "distilbert-base-uncased"
    
    # Performance Settings
    MAX_AUDIO_DURATION = 10
    CONFIDENCE_THRESHOLD = 0.6
    RESPONSE_TIMEOUT = 5