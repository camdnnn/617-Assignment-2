"""Model that uses both image and text to make a class prediction."""

from typing import Tuple

import torch
import torch.nn as nn

from .encoders import (
    ImageEncoderModule,
    TextEncoderModule,
    build_image_encoder,
    build_text_encoder,
)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        num_classes: int = 4,
        dropout: float = 0.2,
        image_encoder_name: str = "convnext_tiny",
        text_encoder_name: str = "distilbert",
    ):
        super().__init__()
        # Pick which image/text models to use from the names passed in.
        self.image_encoder: ImageEncoderModule = build_image_encoder(image_encoder_name)
        self.text_encoder: TextEncoderModule = build_text_encoder(
            text_encoder_name, text_model_name
        )
        # Join image + text features so the final layers can use both.
        dim: int = self.image_encoder.out_dim + self.text_encoder.out_dim
        # This small head turns the combined features into class scores.
        self.head = nn.Sequential(
            # Normalize first so training is more stable across encoder choices.
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def set_encoder_trainable(self, trainable: bool) -> None:
        # Turn both encoders on or off for training in one place.
        for enc in (self.image_encoder, self.text_encoder):
            if trainable:
                enc.train()
            else:
                # Keep encoder behavior fixed when we are not updating them.
                enc.eval()
            for p in enc.parameters():
                p.requires_grad = trainable

    def encode(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return image and text features separately before combining.
        return (
            self.image_encoder(pixel_values),
            self.text_encoder(input_ids=input_ids, attention_mask=attention_mask),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        img_emb, txt_emb = self.encode(pixel_values, input_ids, attention_mask)
        # Combine features for each sample, then output class logits.
        return self.head(torch.cat([img_emb, txt_emb], dim=1))
