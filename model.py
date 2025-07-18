import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DialogueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Keep original text with spaces
        label = self.labels[idx]
        
        # Tokenize the text 
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(file_path):

    df = pd.read_csv(file_path)
    
    # Create label mapping
    unique_labels = df['Label'].unique()
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    # Convert labels to numeric
    df['label_id'] = df['Label'].map(label_to_id)
    
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Labels: {unique_labels}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")
    
    return df, label_to_id, id_to_label

def create_data_loaders(df, tokenizer, batch_size=16, max_length=128):   

    # Split the data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Text'].tolist(),
        df['label_id'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label_id'] # equal distribution of classes in split 
    )
    
    # Create datasets
    train_dataset = DialogueDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = DialogueDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, scheduler, device):

    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions/total_samples:.4f}'
        })
    
    return total_loss / len(train_loader), correct_predictions / total_samples

def evaluate(model, val_loader, device):

    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            
            total_loss += loss.item()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct_predictions / total_samples
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy, all_predictions, all_labels

def main():
    # Configuration
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Load data
    df, label_to_id, id_to_label = load_and_preprocess_data('data.csv')
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        df, tokenizer, BATCH_SIZE, MAX_LENGTH
    )
    
    # Initialize model
    num_labels = len(label_to_id)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels
    )
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # Validation
        val_loss, val_acc, val_predictions, val_labels = evaluate(
            model, val_loader, device
        )
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Best Validation Accuracy: {max(val_accuracies):.4f}")
    
    # Classification report
    label_names = [id_to_label[i] for i in range(len(id_to_label))]
    print("\nClassification Report:")
    print(classification_report(val_labels, val_predictions, target_names=label_names))
    
    # Save the model
    model.save_pretrained('./bert_dialogue_classifier')
    tokenizer.save_pretrained('./bert_dialogue_classifier')
    
    # Save label mappings
    import json
    with open('./bert_dialogue_classifier/label_mappings.json', 'w') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label
        }, f)

    return model, tokenizer, label_to_id, id_to_label