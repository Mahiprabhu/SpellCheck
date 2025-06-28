import torch
import torch.nn as nn
from transformers import BertForMaskedLM

class PhysicsSpellcheckLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

class SpellcheckCNN(nn.Module):
    def __init__(self, vocab_size, char_vocab_size=128, embedding_dim=64):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, vocab_size)
    
    def forward(self, x):
        x = self.char_embedding(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=2)
        return self.fc(x)

def load_pretrained_bert():
    return BertForMaskedLM.from_pretrained('bert-base-uncased')