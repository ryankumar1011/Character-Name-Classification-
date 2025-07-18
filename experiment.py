import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import numpy as np

def load_model(model_path='./bert_dialogue_classifier'):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded")
    
    # Load the model
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded")
    
    # Load label mappings
    with open(f'{model_path}/label_mappings.json', 'r') as f:
        mappings = json.load(f)
        label_to_id = mappings['label_to_id']
        id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
    print("Label mappings loaded")
    
    print(f"Available characters: {list(label_to_id.keys())}")
    
    return model, tokenizer, id_to_label, device

def predict_text(text, model, tokenizer, id_to_label, device, max_length=128):

    print("Input text:", text)

    # Tokenize the input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get character name
    predicted_character = id_to_label[predicted_class]
    
    # Get all probabilities
    all_probs = {}
    for i, prob in enumerate(probabilities[0]):
        character = id_to_label[i]
        all_probs[character] = prob.item()
    
    # Print results
    print(f"Predicted Character: {predicted_character}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll probabilities:")
    for char, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {char}: {prob:.4f}")
    
    return predicted_character, confidence, all_probs

def interactive_test():
    print("Enter text to classify:")
    
    # load model
    model, tokenizer, id_to_label, device = load_model('./bert_dialogue_classifier')
    
    while True:
        user_input = input("\nEnter text: ")
        
        if user_input.lower().strip() in ['quit', 'exit', 'q']:
            break
        
        predict_text(user_input, model, tokenizer, id_to_label, device)

interactive_test()