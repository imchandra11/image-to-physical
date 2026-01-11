import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # batch_H: [batch_size, seq_len, input_size]
        # prev_hidden: (h, c) each [batch_size, hidden_size]
        batch_H_proj = self.i2h(batch_H)  # [batch, seq_len, hidden_size]
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)  # [batch, 1, hidden_size]
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # [batch, seq_len, 1]
        alpha = F.softmax(e, dim=1)  # attention weights
        context = torch.bmm(alpha.permute(0,2,1), batch_H).squeeze(1)  # [batch, input_size]
        concat = torch.cat([context, char_onehots], 1)  # [batch, input_size+num_embeddings]
        cur_hidden = self.rnn(concat, prev_hidden)
        return cur_hidden, alpha

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.generator = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def _char_to_onehot(self, input_char, onehot_dim=None):
        # Converts a batch of indices to one-hot vectors
        batch_size = input_char.size(0)
        if onehot_dim is None:
            onehot_dim = self.num_classes
        one_hot = torch.zeros(batch_size, onehot_dim, device=input_char.device)
        one_hot = one_hot.scatter_(1, input_char.unsqueeze(1), 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] token
        hidden = (
            torch.zeros(batch_size, self.hidden_size, device=batch_H.device),
            torch.zeros(batch_size, self.hidden_size, device=batch_H.device)
        )
        output_hiddens = torch.zeros(batch_size, num_steps, self.hidden_size, device=batch_H.device)

        if is_train:
            # Teacher forcing: feed ground-truth character one-hot at each step
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                hidden, _ = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # hidden state
            probs = self.generator(output_hiddens)  # [batch, num_steps, num_classes]
        else:
            # Inference: use previous predictions as next input
            preds = torch.zeros(batch_size, num_steps, self.num_classes, device=batch_H.device)
            targets = torch.zeros(batch_size, dtype=torch.long, device=batch_H.device)  # start with [GO]=0
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, _ = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])  # [batch, num_classes]
                preds[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input
            probs = preds
        return probs  # [batch_size, num_steps, num_classes]
