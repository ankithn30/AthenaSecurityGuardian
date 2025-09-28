from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import json
import os
from sklearn.model_selection import train_test_split
import os
import sys
import numpy as np
from collections import Counter

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import Config
from src.intent_data import INTENT_DATA, augment_training_data

def compute_class_weights(labels):
    """Compute class weights based on class frequencies"""
    counter = Counter(labels)
    total = sum(counter.values())
    class_weights = {label: total / (len(counter) * count) for label, count in counter.items()}
    max_weight = max(class_weights.values())
    # Normalize weights to prevent any class from having too much influence
    return {label: weight / max_weight for label, weight in class_weights.items()}

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class IntentClassifier:
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self.tokenizer = None
            self.model = None
        
        self.intent_labels = list(INTENT_DATA.keys())
        self.label_map = {intent: idx for idx, intent in enumerate(self.intent_labels)}
        
        if self.model:
            self.model.eval()
    
    def prepare_training_data(self):
        """Prepare training data"""
        data = augment_training_data()
        texts, labels = [], []
        
        for intent, examples in data.items():
            for example in examples:
                texts.append(example)
                labels.append(self.label_map[intent])
        
        return train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    def train_model(self):
        """Train the intent classification model"""
        print("Preparing training data...")
        train_texts, val_texts, train_labels, val_labels = self.prepare_training_data()
        
        print("Initializing model...")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(self.intent_labels)
        )
        
        # Create datasets
        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer)
        
        # Training arguments
        # Compute class weights
        class_weights = compute_class_weights(train_labels)
        print("\nClass weights:")
        for intent, weight in class_weights.items():
            print(f"  {intent}: {weight:.2f}")

        # Convert class weights to tensor
        class_weights_tensor = torch.tensor(
            [class_weights[i] for i in range(len(self.intent_labels))],
            dtype=torch.float
        )
        
        training_args = TrainingArguments(
            output_dir=os.path.join(Config.MODELS_DIR, "intent_classifier", "training"),
            num_train_epochs=5,  # Increased epochs for better learning
            per_device_train_batch_size=8,  # Smaller batch size for better generalization
            per_device_eval_batch_size=8,
            learning_rate=1e-5,  # Lower learning rate for more stable training
            weight_decay=0.01,
            warmup_ratio=0.1,  # Add warmup to stabilize training
            logging_dir='./logs',
            logging_steps=10,
            save_steps=100,  # Save every 100 steps
            eval_steps=100,  # Evaluate every 100 steps
            report_to="none"  # Disable reporting to prevent wandb errors
        )
        
        # Define compute metrics function for evaluation
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            
            # Calculate accuracy
            accuracy = (preds == labels).mean()
            
            # Calculate F1 score (weighted average)
            f1_scores = []
            for label in range(len(self.intent_labels)):
                true_pos = ((preds == label) & (labels == label)).sum()
                false_pos = ((preds == label) & (labels != label)).sum()
                false_neg = ((preds != label) & (labels == label)).sum()
                
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
            
            avg_f1 = sum(f1_scores) / len(f1_scores)
            
            return {
                'accuracy': accuracy,
                'f1': avg_f1
            }
        
        # Initialize trainer with compute_metrics
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # Set loss weights in the model
        self.model.class_weights = class_weights_tensor
        
        print("Training model...")
        trainer.train()
        
        # Save model
        model_path = os.path.join(Config.MODELS_DIR, "intent_classifier", "model")
        os.makedirs(model_path, exist_ok=True)
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        print(f"Model saved to {model_path}")
    
    def classify_intent(self, text):
        """Classify intent of input text"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            "intent": self.intent_labels[predicted_class],
            "confidence": confidence
        }

# Training script
if __name__ == "__main__":
    classifier = IntentClassifier()
    classifier.train_model()
    print("Intent classifier training complete!")