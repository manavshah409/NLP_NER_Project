"""Linear-Chain Conditional Random Field (CRF) PyTorch module.

Supports forward log-partition calculation, gold sequence scoring,
exact Negative Log-Likelihood (NLL) loss, and Viterbi decoding with sequence masking.
"""
from typing import List, Optional
import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """Linear-chain CRF layer for sequence labeling."""

    def __init__(self, num_tags: int):
        super().__init__()
        if num_tags <= 0:
            raise ValueError(f"num_tags must be positive, got {num_tags}")
        self.num_tags = num_tags

        # transitions[i, j] = score of transitioning from tag i to tag j (i -> j)
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize transition parameters uniformly."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute Negative Log-Likelihood (NLL) loss: log Z(x) - Score(x, y).

        Args:
            emissions: Tensor of shape (batch_size, seq_len, num_tags).
            tags: LongTensor of shape (batch_size, seq_len) with target tag indices.
            mask: BoolTensor of shape (batch_size, seq_len), 1 for valid tokens, 0 for pad.
            reduction: 'mean', 'sum', or 'none'.

        Returns:
            Scalar loss tensor (or tensor of shape (batch_size,) if reduction='none').
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()

        log_partition = self._compute_log_partition(emissions, mask)
        gold_score = self._compute_gold_score(emissions, tags, mask)
        nll = log_partition - gold_score

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        elif reduction == "none":
            return nll
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the unnormalized score for target tag sequences.

        Score = start_trans[y_0] + sum_t emissions[t, y_t] + sum_{t>0} transitions[y_{t-1}, y_t] + end_trans[y_last]
        """
        batch_size, seq_len, _ = emissions.shape
        # Score starts at t=0
        first_tags = tags[:, 0]
        score = self.start_transitions[first_tags] + emissions[torch.arange(batch_size, device=emissions.device), 0, first_tags]

        # Accumulate transitions and emissions for t=1..seq_len-1
        for t in range(1, seq_len):
            mask_t = mask[:, t]
            prev_tags = tags[:, t - 1]
            curr_tags = tags[:, t]

            # Transition score from prev_tag to curr_tag
            trans_score = self.transitions[prev_tags, curr_tags]
            # Emission score for curr_tag
            emit_score = emissions[torch.arange(batch_size, device=emissions.device), t, curr_tags]

            step_score = trans_score + emit_score
            score = score + torch.where(mask_t, step_score, torch.zeros_like(score))

        # Add end transitions at the last valid index for each sequence
        # seq_lens gives number of valid tokens (at least 1)
        seq_lens = mask.long().sum(dim=1)  # shape (batch_size,)
        last_indices = (seq_lens - 1).clamp(min=0)
        last_tags = tags.gather(1, last_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]

        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log Z(x) using the forward algorithm in log-space."""
        batch_size, seq_len, num_tags = emissions.shape

        # alpha_0 = start_transitions + emissions[:, 0] -> shape (batch_size, num_tags)
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        for t in range(1, seq_len):
            mask_t = mask[:, t].unsqueeze(1)  # shape (batch_size, 1)

            # alpha: (batch_size, num_tags, 1)
            # transitions: (1, num_tags, num_tags) where [0, i, j] is i -> j
            # emit_t: (batch_size, 1, num_tags)
            emit_t = emissions[:, t].unsqueeze(1)
            broadcast = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t
            # logsumexp over dim=1 (previous tags i)
            next_alpha = torch.logsumexp(broadcast, dim=1)  # shape (batch_size, num_tags)

            # Masked update: keep old alpha where masked out
            alpha = torch.where(mask_t, next_alpha, alpha)

        # Add end transitions and marginalize over final tags
        terminal_alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(terminal_alpha, dim=1)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Find the Viterbi path (highest-scoring tag sequence) for each sample.

        Args:
            emissions: Tensor of shape (batch_size, seq_len, num_tags).
            mask: BoolTensor of shape (batch_size, seq_len).

        Returns:
            List of length `batch_size`, each containing the list of tag IDs of length `seq_len_i`.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()

        batch_size, seq_len, num_tags = emissions.shape
        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (batch_size, num_tags)
        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            mask_t = mask[:, t].unsqueeze(1)  # (batch_size, 1)

            # (batch_size, num_tags, 1) + (1, num_tags, num_tags) -> (batch_size, num_tags, num_tags)
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_scores, best_tags = scores.max(dim=1)  # (batch_size, num_tags)

            next_viterbi = max_scores + emissions[:, t]
            backpointers.append(best_tags)
            viterbi = torch.where(mask_t, next_viterbi, viterbi)

        terminal_scores = viterbi + self.end_transitions.unsqueeze(0)
        best_last_tags = terminal_scores.argmax(dim=1).tolist()  # list of length batch_size

        seq_lens = mask.long().sum(dim=1).tolist()
        best_paths: List[List[int]] = []

        for b in range(batch_size):
            length = seq_lens[b]
            if length <= 0:
                best_paths.append([])
                continue

            last_tag = best_last_tags[b]
            path = [last_tag]

            # Backtrack from step length - 2 down to 0
            curr_tag = last_tag
            for t in range(length - 2, -1, -1):
                curr_tag = backpointers[t][b, curr_tag].item()
                path.append(curr_tag)

            path.reverse()
            best_paths.append(path)

        return best_paths
