import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intent_classifier import IntentClassifier
from src.config import Config

def test_intent_classification():
    # Load trained model
    model_path = os.path.join(Config.MODELS_DIR, "intent_classifier", "model")
    classifier = IntentClassifier(model_path)
    
    test_phrases = [
        "Hello there",
        "Where is room 203?",
        "I need access to the building",
        "This is an emergency",
        "I'm here to see John Smith",
        "What are the building hours?",
        "How's the weather today?",
        "Good morning, can I get some assistance?",
        "Need help getting to the conference room",
        "Can I get my security badge activated?",
        "There's smoke in the lobby!",
        "I have a meeting with Sarah from accounting",
        "Is the cafeteria open now?",
        "Just looking around"
    ]
    
    print("Testing Intent Classification:")
    print("-" * 40)
    
    for phrase in test_phrases:
        result = classifier.classify_intent(phrase)
        print(f"Text: '{phrase}'")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        print()

if __name__ == "__main__":
    test_intent_classification()