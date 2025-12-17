import torch
import torch.nn as nn
import timm

class SteeringViT(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained Vision Transformer
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0   # removes classification head
        )

        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.vit(x)          # (B, 768)
        x = self.regressor(x)   # (B, 1)
        return x.squeeze(1)
