import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, num_layers=2,
                 dropout=0.5, activation="tanh", bidirectional=False, model_type="rnn"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.num_directions = 2 if bidirectional else 1
        if model_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                              dropout=dropout, batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                               dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        embedded = self.embed(x)
        output, hidden = self.rnn(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last_hidden = hidden[-self.num_directions:].transpose(0,1).reshape(x.size(0), -1)
        if self.activation == "relu":
            last_hidden = F.relu(last_hidden)
        elif self.activation == "sigmoid":
            last_hidden = torch.sigmoid(last_hidden)
        else:
            last_hidden = torch.tanh(last_hidden)
        out = self.fc(self.dropout(last_hidden))
        return out.squeeze(1)

class RNNClassifier(BaseTextModel):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, model_type="rnn", **kwargs)

class LSTMClassifier(BaseTextModel):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, model_type="lstm", **kwargs)

class BiLSTMClassifier(BaseTextModel):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, model_type="lstm", bidirectional=True, **kwargs)
