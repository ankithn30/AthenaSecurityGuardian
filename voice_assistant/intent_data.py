"""
Intent Data for Athena Security Guardian
Training data for security-specific intent classification
"""

import json
import os
from pathlib import Path

# Security-specific training data for intent classification
INTENT_DATA = {
    "greeting": [
        "hello", "hi there", "good morning", "good afternoon", "good evening",
        "hey", "hello athena", "hi athena", "greetings", "good day",
        "hello there", "hey there", "hi good morning", "good evening athena",
        "hello security", "hi security assistant", "greetings athena"
    ],
    
    "direction_query": [
        "where is room 203", "how do I get to room 101", "where can I find the bathroom",
        "directions to the cafeteria", "location of room 205", "where is the elevator",
        "how to get to parking", "where is the exit", "find meeting room A",
        "where is room 301", "directions to the library", "how do I find room B12",
        "where are the restrooms", "location of conference room", "where is the main entrance",
        "how to get to the second floor", "where is the emergency exit",
        "find the visitor parking", "directions to reception", "where is the security office",
        "how do I get to building C", "location of the auditorium", "where is room 150",
        "find the nearest bathroom", "directions to the stairwell", "where is the copy room",
        "how to reach the basement", "location of room 404", "where is the break room"
    ],
    
    "access_request": [
        "can I enter", "I need access to building", "open the door please",
        "let me in", "I'm authorized to enter", "I have permission",
        "unlock the door", "access granted", "security clearance",
        "I need to get in", "please open", "I have authorization",
        "can you unlock this", "I'm cleared for entry", "building access needed",
        "door access please", "I need entry", "authorized personnel",
        "I have my badge", "security access", "I'm on the list",
        "employee access", "staff entry", "I work here", "I have clearance"
    ],
    
    "emergency": [
        "help me", "emergency", "call security", "there's a problem",
        "urgent help needed", "assistance required", "security emergency",
        "medical emergency", "fire alarm", "suspicious activity",
        "help", "urgent", "emergency situation", "need immediate help",
        "call for help", "security alert", "danger", "threat detected",
        "medical assistance", "fire emergency", "evacuation needed",
        "intruder alert", "security breach", "panic", "crisis",
        "someone is hurt", "accident", "need paramedics", "call police",
        "suspicious person", "security incident", "immediate assistance"
    ],
    
    "visitor_inquiry": [
        "I'm here to see John", "I have an appointment", "visiting someone",
        "meeting with manager", "scheduled appointment", "here for interview",
        "delivery for office", "contractor reporting", "visitor check in",
        "I'm here to meet", "appointment with", "visiting employee",
        "here for a meeting", "scheduled visit", "business appointment",
        "here to see the director", "meeting scheduled", "visitor registration",
        "I have a visitor badge", "checking in", "here for consultation",
        "appointment at 2pm", "meeting with team", "here for presentation",
        "visitor access", "guest registration", "here to meet client"
    ],
    
    "facility_info": [
        "what time does building close", "building hours", "when do you open",
        "operating hours", "wifi password", "parking information",
        "visitor policy", "security procedures", "building rules",
        "what are the hours", "when does it close", "opening times",
        "facility hours", "business hours", "access hours",
        "wifi network", "internet access", "guest wifi",
        "parking rules", "visitor parking", "parking fees",
        "building policies", "security protocols", "safety procedures",
        "facility information", "building guidelines", "office hours",
        "weekend hours", "holiday schedule", "after hours access"
    ],
    
    "unknown": [
        "what's the weather", "how are you", "tell me a joke",
        "random question", "unrelated query", "personal question",
        "weather forecast", "sports scores", "news updates",
        "what's your name", "how old are you", "where are you from",
        "favorite color", "do you like music", "what's for lunch",
        "stock prices", "movie recommendations", "book suggestions",
        "travel advice", "cooking recipes", "shopping list",
        "game scores", "celebrity news", "entertainment"
    ]
}

# Response templates for each intent type
RESPONSES = {
    'greeting': [
        'Hello! Welcome to the security assistance system. How may I help you?',
        'Hi there! I\'m here to assist with security and facility matters.',
        'Greetings! How can I help you today?'
    ],
    'direction_query': [
        'Let me help you find that location.',
        'I can guide you to that destination.',
        'Here are the directions you need.'
    ],
    'access_request': [
        'Please present your credentials for verification.',
        'I\'ll need to verify your access credentials.',
        'One moment while I check your authorization.'
    ],
    'emergency': [
        'I\'m alerting security personnel immediately.',
        'Emergency services have been notified.',
        'Help is on the way. Please remain calm.'
    ],
    'visitor_inquiry': [
        'Let me check the visitor registry for you.',
        'I\'ll help you with the check-in process.',
        'Please provide your appointment details.'
    ],
    'facility_info': [
        'Here\'s the facility information you requested.',
        'Let me provide you with that building information.',
        'I have that facility information for you.'
    ],
    'unknown': [
        'I apologize, but I can only assist with security and facility-related matters.',
        'That\'s outside my scope. I can help with security, access, and building information.',
        'I\'m focused on security assistance. Please ask about building access, directions, or facility information.'
    ]
}

# Additional data for data augmentation
INTENT_EXPANSIONS = {
    "greeting": {
        "prefixes": ["", "excuse me", "hello there", "good morning"],
        "suffixes": ["", "please", "thank you", "can you help me"]
    },
    "direction_query": {
        "prefixes": ["excuse me", "can you tell me", "I need to know", "please help"],
        "suffixes": ["please", "thank you", "if you know", "I'm lost"]
    },
    "access_request": {
        "prefixes": ["excuse me", "hello", "I need", "can you please"],
        "suffixes": ["please", "I'm authorized", "I have permission", "thank you"]
    },
    "emergency": {
        "prefixes": ["", "urgent", "immediate", "please"],
        "suffixes": ["now", "immediately", "right away", "please help"]
    }
}

def get_config():
    """Get configuration paths"""
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    TRAINING_DIR = DATA_DIR / "training"
    
    return {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR, 
        "training_dir": TRAINING_DIR
    }

def create_directories():
    """Create necessary directories"""
    config = get_config()
    config["data_dir"].mkdir(exist_ok=True)
    config["training_dir"].mkdir(exist_ok=True)
    print(f"✓ Created directories: {config['training_dir']}")

def augment_training_data():
    """Augment training data with variations"""
    print("🔄 Augmenting training data...")
    augmented = {}
    
    for intent, examples in INTENT_DATA.items():
        augmented[intent] = examples.copy()
        
        # Add variations based on expansions
        if intent in INTENT_EXPANSIONS:
            prefixes = INTENT_EXPANSIONS[intent]["prefixes"]
            suffixes = INTENT_EXPANSIONS[intent]["suffixes"]
            
            for example in examples[:5]:  # Only augment first 5 to avoid explosion
                for prefix in prefixes[:2]:  # Limit combinations
                    for suffix in suffixes[:2]:
                        if prefix or suffix:
                            new_example = f"{prefix} {example} {suffix}".strip()
                            if new_example not in augmented[intent]:
                                augmented[intent].append(new_example)
        
        # Add question variations for direction queries
        if intent == "direction_query":
            for example in examples:
                if not example.endswith("?"):
                    augmented[intent].append(example + "?")
        
        # Add polite variations
        if intent in ["access_request", "direction_query"]:
            for example in examples[:3]:  # Limit to avoid too many
                if "please" not in example:
                    augmented[intent].append("please " + example)
    
    # Print augmentation statistics
    for intent in augmented:
        original = len(INTENT_DATA[intent])
        augmented_count = len(augmented[intent])
        print(f"  {intent}: {original} → {augmented_count} examples")
    
    return augmented

def save_training_data():
    """Save training data to JSON file"""
    print("💾 Saving training data...")
    config = get_config()
    
    # Save original data
    original_file = config["training_dir"] / "intent_data_original.json"
    with open(original_file, "w", encoding="utf-8") as f:
        json.dump(INTENT_DATA, f, indent=2, ensure_ascii=False)
    
    # Save augmented data
    augmented_data = augment_training_data()
    augmented_file = config["training_dir"] / "intent_data_augmented.json"
    with open(augmented_file, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    
    # Create labels file
    labels = list(INTENT_DATA.keys())
    labels_file = config["training_dir"] / "intent_labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    
    # Create statistics
    stats = {
        "total_intents": len(INTENT_DATA),
        "intent_counts": {intent: len(examples) for intent, examples in INTENT_DATA.items()},
        "augmented_counts": {intent: len(examples) for intent, examples in augmented_data.items()},
        "total_original_examples": sum(len(examples) for examples in INTENT_DATA.values()),
        "total_augmented_examples": sum(len(examples) for examples in augmented_data.values())
    }
    
    stats_file = config["training_dir"] / "training_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Original data saved: {original_file}")
    print(f"✓ Augmented data saved: {augmented_file}")
    print(f"✓ Labels saved: {labels_file}")
    print(f"✓ Statistics saved: {stats_file}")
    
    return stats

def load_training_data(use_augmented=True):
    """Load training data from file"""
    config = get_config()
    
    if use_augmented:
        file_path = config["training_dir"] / "intent_data_augmented.json"
    else:
        file_path = config["training_dir"] / "intent_data_original.json"
    
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️ Training data file not found: {file_path}")
        print("Using default INTENT_DATA")
        return INTENT_DATA

def get_intent_patterns():
    """Returns all intent patterns for training"""
    return INTENT_DATA

def get_response(intent, **kwargs):
    """Returns a random response for the given intent"""
    import random
    responses = RESPONSES.get(intent, RESPONSES['unknown'])
    response = random.choice(responses)
    return response.format(**kwargs) if kwargs else response

def print_data_summary():
    """Print summary of training data"""
    print("\n" + "="*60)
    print("ATHENA SECURITY GUARDIAN - INTENT DATA SUMMARY")
    print("="*60)
    
    total_examples = 0
    for intent, examples in INTENT_DATA.items():
        count = len(examples)
        total_examples += count
        print(f"{intent:20}: {count:3d} examples")
    
    print("-"*40)
    print(f"{'Total':20}: {total_examples:3d} examples")
    print(f"{'Total Intents':20}: {len(INTENT_DATA):3d}")
    
    print("\n📝 Example phrases:")
    for intent, examples in INTENT_DATA.items():
        print(f"\n{intent.upper()}:")
        for example in examples[:3]:  # Show first 3 examples
            print(f"  • {example}")
        if len(examples) > 3:
            print(f"  • ... and {len(examples) - 3} more")

def validate_data():
    """Validate training data quality"""
    print("\n🔍 Validating training data...")
    
    issues = []
    
    # Check for empty intents
    for intent, examples in INTENT_DATA.items():
        if not examples:
            issues.append(f"Empty intent: {intent}")
        elif len(examples) < 5:
            issues.append(f"Low example count for {intent}: {len(examples)}")
    
    # Check for duplicate examples
    all_examples = []
    for intent, examples in INTENT_DATA.items():
        for example in examples:
            if example in all_examples:
                issues.append(f"Duplicate example found: '{example}'")
            all_examples.append(example)
    
    # Check for very short examples
    for intent, examples in INTENT_DATA.items():
        for example in examples:
            if len(example.split()) < 2 and intent != "greeting":
                issues.append(f"Very short example in {intent}: '{example}'")
    
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✅ Data validation passed!")
    
    return len(issues) == 0

def main():
    """Main function to setup intent data"""
    print("🚀 Setting up Intent Data for Athena Security Guardian")
    print("="*60)
    
    try:
        # Create directories
        create_directories()
        
        # Validate data
        if not validate_data():
            print("❌ Data validation failed. Please fix issues before proceeding.")
            return False
        
        # Print summary
        print_data_summary()
        
        # Save data
        stats = save_training_data()
        
        print("\n" + "="*60)
        print("📊 TRAINING DATA STATISTICS")
        print("="*60)
        print(f"Total Intents: {stats['total_intents']}")
        print(f"Original Examples: {stats['total_original_examples']}")
        print(f"Augmented Examples: {stats['total_augmented_examples']}")
        print(f"Data Augmentation: +{stats['total_augmented_examples'] - stats['total_original_examples']} examples")
        
        print("\n✅ Intent data setup completed successfully!")
        print("\n📁 Files created:")
        config = get_config()
        training_dir = config["training_dir"]
        print(f"  • {training_dir}/intent_data_original.json")
        print(f"  • {training_dir}/intent_data_augmented.json")
        print(f"  • {training_dir}/intent_labels.json")
        print(f"  • {training_dir}/training_statistics.json")
        
        print("\n🎯 Ready for intent classifier training!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up intent data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ You can now proceed to the next step: python tests/test_audio.py")
    else:
        print("\n❌ Please fix the issues above before continuing.")