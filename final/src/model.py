import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True)

    def forward(self, input):
        output, hidden = self.gru(input, None)
        return output, hidden

class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True)
        self.embed = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, 302)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, enc_out):
        embed = self.embed(input)
        attn_weights = F.softmax(self.attn(torch.cat((embed.squeeze(), hidden.squeeze()), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        output = torch.cat((embed, attn_applied), -1)
        output = self.attn_combine(output)
        output, hidden = self.gru(output, hidden)
        output = self.output(output)
        return output, hidden
