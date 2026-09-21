"""BiLSTM-CRF PyTorch model for Named Entity Recognition."""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from src.models.linear_chain_crf import LinearChainCRF

# Canonical 7-class BIO schema
LABELS = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABELS)}
NUM_TAGS = len(LABELS)


class BiLSTM_CRF(nn.Module):
    """Bidirectional LSTM with Linear-Chain CRF for sequence tagging."""

    def __init__(
        self,
        vocab_size: int,
        num_tags: int = NUM_TAGS,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        padding_idx: int = 0,
        use_crf: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        self.padding_idx = padding_idx
        self.use_crf = use_crf

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.emission_linear = nn.Linear(lstm_output_dim, num_tags)

        if use_crf:
            self.crf = LinearChainCRF(num_tags=num_tags)
        else:
            self.crf = None

    def get_emissions(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute tag emission scores for input token IDs.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len).
            mask: Optional BoolTensor of shape (batch_size, seq_len).

        Returns:
            Emissions tensor of shape (batch_size, seq_len, num_tags).
        """
        # (batch_size, seq_len, embedding_dim)
        embeds = self.dropout(self.embedding(input_ids))

        # Pack padded sequence if mask is provided for faster LSTM compute
        if mask is not None:
            lengths = mask.long().sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=input_ids.size(1)
            )
        else:
            lstm_out, _ = self.lstm(embeds)

        lstm_out = self.dropout(lstm_out)
        emissions = self.emission_linear(lstm_out)
        return emissions

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        If tags are provided and CRF is enabled, returns scalar NLL loss.
        Otherwise, returns emissions tensor.
        """
        emissions = self.get_emissions(input_ids, mask=mask)

        if tags is not None and self.crf is not None:
            return self.crf(emissions, tags, mask=mask, reduction=reduction)
        return emissions

    def decode(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Decode optimal tag paths for a batch of sequences."""
        emissions = self.get_emissions(input_ids, mask=mask)
        if self.crf is not None:
            return self.crf.decode(emissions, mask=mask)
        # Softmax greedy decode fallback if no CRF
        preds = emissions.argmax(dim=-1).tolist()
        if mask is not None:
            lengths = mask.long().sum(dim=1).tolist()
            return [preds[b][: lengths[b]] for b in range(len(lengths))]
        return preds
