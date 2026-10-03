"""SegFormer B0/B1 encoder with an explicit full-resolution detail decoder."""

from __future__ import annotations


def _variant_config(variant: str) -> dict:
    if variant == "b0":
        return {"hidden_sizes": [32, 64, 160, 256], "depths": [2, 2, 2, 2]}
    if variant == "b1":
        return {"hidden_sizes": [64, 128, 320, 512], "depths": [2, 2, 2, 2]}
    raise ValueError(f"unsupported SegFormer variant: {variant}")


def create_segformer_detail(variant: str = "b1", pretrained: bool = False,
                            pretrained_name: str | None = None):
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from transformers import SegformerConfig, SegformerModel
    except ImportError as error:
        raise RuntimeError("SegFormer candidates require torch and transformers") from error

    settings = _variant_config(variant)
    checkpoint = pretrained_name or f"nvidia/mit-{variant}"
    if pretrained:
        encoder = SegformerModel.from_pretrained(checkpoint)
        hidden_sizes = list(encoder.config.hidden_sizes)
    else:
        config = SegformerConfig(
            num_channels=3,
            num_encoder_blocks=4,
            depths=settings["depths"],
            hidden_sizes=settings["hidden_sizes"],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            sr_ratios=[8, 4, 2, 1],
        )
        encoder = SegformerModel(config)
        hidden_sizes = settings["hidden_sizes"]

    class ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            groups = 8 if output_channels % 8 == 0 else 4
            self.body = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels),
                nn.GELU(),
            )

        def forward(self, value):
            return self.body(value)

    class SegFormerDetail(nn.Module):
        model_type = f"segformer_{variant}_detail"

        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            decoder_channels = 128
            self.projections = nn.ModuleList(
                nn.Conv2d(channels, decoder_channels, 1) for channels in hidden_sizes)
            self.fuse = ConvBlock(decoder_channels * 4, decoder_channels)
            self.detail = ConvBlock(3, 32)
            self.full = ConvBlock(decoder_channels + 32, 64)
            self.mask_head = nn.Conv2d(64, 1, 1)
            self.distance_head = nn.Conv2d(64, 1, 1)
            self.auxiliary_head = nn.Conv2d(decoder_channels, 1, 1)

        def set_encoder_trainable(self, trainable: bool) -> None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(trainable)

        def forward(self, value):
            output = self.encoder(value, output_hidden_states=True, return_dict=True)
            features = output.hidden_states[-4:]
            target_size = features[0].shape[-2:]
            projected = []
            for index, (feature, projection) in enumerate(
                    zip(features, self.projections)):
                current = projection(feature)
                # The MiT stages after the first are always lower resolution.
                # Using the known stage index keeps the ONNX trace free of a
                # tensor-to-Python-boolean warning without changing outputs.
                if index > 0:
                    current = F.interpolate(current, size=target_size, mode="bilinear",
                                            align_corners=False)
                projected.append(current)
            fused = self.fuse(torch.cat(projected, dim=1))
            auxiliary = self.auxiliary_head(fused)
            full = F.interpolate(fused, size=value.shape[-2:], mode="bilinear",
                                 align_corners=False)
            full = self.full(torch.cat((full, self.detail(value)), dim=1))
            return {
                "mask_logits": self.mask_head(full),
                "distance": torch.tanh(self.distance_head(full)),
                "auxiliary_logits": auxiliary,
            }

    return SegFormerDetail()
