import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.crf = CRF(output_size, batch_first=True)

    def forward(self, x, tags):
        """
        x: (batch_size, seq_len)
        tags: (batch_size, seq_len)
        """
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)
        x = self.linear(x)  # (batch_size, seq_len, output_size)
        preds = torch.tensor(self.crf.decode(x))  # (batch_size, seq_len)
        loss = -1 * self.crf(x, tags, reduction="token_mean")  # ()

        return loss, preds
