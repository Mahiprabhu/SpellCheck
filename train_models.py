import os
import torch
from torch import nn
from torch.optim import AdamW
import json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from models import PhysicsSpellcheckLSTM, SpellcheckCNN, load_pretrained_bert
from config import Config
import torch.optim as optim

config = Config()

# Common dataset and vocabulary handling
class PhysicsDataset:
    @staticmethod
    def load_terms():
        with open(os.path.join(config.DATA_DIR, 'physics_terms.txt'), 'r') as f:
            terms = [line.strip().lower() for line in f if line.strip()]
        return ['<PAD>', '<UNK>'] + list(OrderedDict.fromkeys(terms))

    @staticmethod
    def build_vocab(terms):
        vocab = {word: idx for idx, word in enumerate(terms)}
        if max(vocab.values()) != len(vocab) - 1:
            raise ValueError("Vocabulary indexing error")
        return vocab

class LSTMDataset(Dataset):
    def __init__(self, vocab):
        self.sequences = []
        terms = list(vocab.keys())[2:]  # Skip special tokens
        for i in range(len(terms)-1):
            self.sequences.append((vocab[terms[i]], vocab[terms[i+1]]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx][0]), torch.tensor(self.sequences[idx][1])

class CNNDataset(Dataset):
    def __init__(self, vocab, max_len=16):
        self.data = []
        self.char_vocab = {'<PAD>': 0, '<UNK>': 1}
        terms = list(vocab.keys())
        
        # Build character vocabulary
        for term in terms:
            for c in term:
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)
        
        for term in terms:
            char_ids = [self.char_vocab.get(c, 1) for c in term[:max_len]]
            char_ids += [0] * (max_len - len(char_ids))
            self.data.append((
                torch.tensor(char_ids, dtype=torch.long),
                torch.tensor(vocab[term], dtype=torch.long)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Training functions
def train_lstm(vocab, save_path):
    dataset = LSTMDataset(vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PhysicsSpellcheckLSTM(len(vocab))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/10"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs.view(-1, len(vocab)), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(dataloader):.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, save_path)

def train_cnn(vocab, save_path):
    dataset = CNNDataset(vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SpellcheckCNN(
        vocab_size=len(vocab),
        char_vocab_size=len(dataset.char_vocab)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        
        for char_ids, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/10"):
            char_ids, targets = char_ids.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(char_ids)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(dataloader):.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_vocab': dataset.char_vocab,
        'word_vocab': vocab
    }, save_path)

def train_bert(vocab, save_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_pretrained_bert()
    
    # Prepare physics terms data
    dataset = []
    terms = list(vocab.keys())[2:]  # Skip special tokens
    for term in tqdm(terms, desc="Preparing BERT data"):
        inputs = tokenizer(term, return_tensors='pt')
        input_ids = inputs['input_ids'][0]
        
        for i in range(1, len(input_ids)-1):  # Skip [CLS] and [SEP]
            masked_ids = input_ids.clone()
            masked_ids[i] = tokenizer.mask_token_id
            dataset.append({
                'input_ids': masked_ids.unsqueeze(0),
                'attention_mask': inputs['attention_mask'][0].unsqueeze(0),
                'labels': input_ids.unsqueeze(0)
            })
    
    # Training setup
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=bert_collate_fn)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(3):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/3"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), save_path)

def bert_collate_fn(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = max(len(item['input_ids'][0]) for item in batch)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Pad input_ids and attention_mask
        pad_len = max_len - len(item['input_ids'][0])
        input_ids.append(
            torch.cat([
                item['input_ids'][0],
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
        )
        attention_masks.append(
            torch.cat([
                item['attention_mask'][0],
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )
        # Pad labels (using -100 for padding)
        labels.append(
            torch.cat([
                item['labels'][0],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
        )
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }

if __name__ == '__main__':
    # Load and validate physics terms
    terms = PhysicsDataset.load_terms()
    vocab = PhysicsDataset.build_vocab(terms)
    
    # Train all models
    print("Training LSTM...")
    train_lstm(vocab, os.path.join(config.DATA_DIR, 'lstm_physics.pth'))
    
    print("\nTraining CNN...")
    train_cnn(vocab, os.path.join(config.DATA_DIR, 'cnn_physics.pth'))
    
    print("\nFine-tuning BERT...")
    train_bert(vocab, os.path.join(config.DATA_DIR, 'bert_finetuned_physics.pth'))