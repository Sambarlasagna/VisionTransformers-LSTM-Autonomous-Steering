import torch
import torch.nn as nn
import timm

class SteeringViTLSTM(nn.Module):
    def __init__(self, sequence_length=5, hidden_dim=256):
        super().__init__()

        # ---- ViT backbone (spatial intelligence) ----
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0   # remove classifier
        )

        vit_embed_dim = 768

        # ---- LSTM (temporal intelligence) ----
        self.lstm = nn.LSTM(
            input_size=vit_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # ---- Regression head ----
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x shape: (B, T, 3, 224, 224)
        """
        B, T, C, H, W = x.shape

        # ---- Merge batch & time for ViT ----
        x = x.view(B * T, C, H, W)

        # ---- Spatial features ----
        features = self.vit(x)          # (B*T, 768)

        # ---- Restore sequence ----
        features = features.view(B, T, -1)  # (B, T, 768)

        # ---- Temporal modeling ----
        lstm_out, _ = self.lstm(features)   # (B, T, hidden_dim)

        # Take last timestep output
        last_out = lstm_out[:, -1, :]       # (B, hidden_dim)

        # ---- Steering regression ----
        steering = self.regressor(last_out) # (B, 1)

        return steering.squeeze(1)
