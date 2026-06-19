from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SequenceModelConfig:
    """
    Hyperparameters for the temporal fusion model.

    The architecture follows guidance from *Deep Learning* (Goodfellow et al., 2016)
    which highlights gated recurrent units for long-range dependencies, while the optional
    attention head provides the interpretability advocated by *AI Engineering* (Amershi et al., 2023).
    """

    input_size: int
    static_size: int
    hidden_size: int = 128
    temporal_layers: int = 2
    dropout: float = 0.2
    attention_heads: int = 4
    feedforward_size: int = 128


class SequenceFusionModel(nn.Module):
    """
    Combine temporal encoders with static feature projections for stock forecasting.

    Temporal signals are processed by a stacked LSTM and, when configured, refined through
    multi-head attention so the model can reweight salient timesteps. Static descriptors
    (fundamentals, recommendation aggregates) flow through a lightweight MLP and are fused
    with the temporal context following practices outlined in the *LLM Engineer's Handbook*
    around multi-modal conditioning.
    """

    def __init__(self, config: SequenceModelConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_layers = max(1, int(config.temporal_layers))
        self.hidden_size = int(config.hidden_size)
        dropout = float(config.dropout)

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.temporal_layers,
            batch_first=True,
            dropout=dropout if self.temporal_layers > 1 else 0.0,
        )

        self.use_attention = int(config.attention_heads) > 0
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=int(config.attention_heads),
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.attention = None

        static_size = max(0, int(config.static_size))
        if static_size > 0:
            self.static_encoder = nn.Sequential(
                nn.LayerNorm(static_size),
                nn.Linear(static_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.static_encoder = None

        ff_size = max(self.hidden_size, int(config.feedforward_size))
        fusion_input = self.hidden_size
        if self.static_encoder is not None:
            fusion_input += self.hidden_size

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, ff_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2 if dropout > 0 else 0.0),
        )
        self.output = nn.Linear(max(1, ff_size // 2), 1)

    def forward(
        self,
        temporal_inputs: torch.Tensor,
        static_inputs: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        temporal_inputs:
            Tensor of shape (batch, seq_len, input_size)
        static_inputs:
            Tensor of shape (batch, static_size) or ``None`` if static features absent.
        return_attention:
            When True, returns the temporal attention weights for interpretability.

        Returns
        -------
        (logits, attn_weights)
        """

        lstm_out, _ = self.lstm(temporal_inputs)
        if self.use_attention and self.attention is not None:
            # Attend over temporal dimension; key/query/value identical for self-attention.
            attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # Aggregate by combining last hidden state with attention pooled summary.
            temporal_summary = torch.cat([attn_output[:, -1, :], lstm_out[:, -1, :]], dim=-1)
            temporal_summary = self._mix_temporal_summary(temporal_summary)
        else:
            attn_weights = None
            temporal_summary = lstm_out[:, -1, :]

        features = temporal_summary
        if self.static_encoder is not None and static_inputs is not None and static_inputs.numel() > 0:
            static_repr = self.static_encoder(static_inputs)
            features = torch.cat([features, static_repr], dim=-1)

        fused = self.fusion(features)
        logits = self.output(fused).squeeze(-1)
        if return_attention:
            return logits, attn_weights
        return logits, None

    def predict_proba(
        self, temporal_inputs: torch.Tensor, static_inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits, _ = self.forward(temporal_inputs, static_inputs, return_attention=False)
        return torch.sigmoid(logits)

    def _mix_temporal_summary(self, combined: torch.Tensor) -> torch.Tensor:
        """
        Reduce doubled hidden dimension after concatenating attention-pooled and terminal states.
        """
        size = combined.shape[-1] // 2
        mix = combined.view(combined.shape[0], 2, size)
        return mix.sum(dim=1) / math.sqrt(2.0)
