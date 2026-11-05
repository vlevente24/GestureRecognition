import torch
import torch.nn as nn
import torchvision.models as models
from backbone import ResNet3D18


class GestureRecognizer(nn.Module):
    """
    Gesture Recognizer model combining 3D ResNet18 backbone and Transformer Encoder.
    
    Parameters
    ----------
    num_classes: int
        Number of output classes for classification

    Features
    --------
    x: 3D feature tensor
        (B, T, C, H, W)

    Returns
    -------
        output: (B, num_classes)
    """

    def __init__(self, num_classes=27, backbone_pretrain=None, transformer_pretrain=None):
        super().__init__()

        # 3D ResNet18 backbone
        self.backbone = ResNet3D18(num_classes=num_classes, classify=False, pretrained=backbone_pretrain)
        self.dropout_backbone = nn.Dropout(p=0.2)
        
        # Transformer Encoder blokk, amely tartalmaz Multihead Attention-t is
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Klasszifikációs fej
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(512, num_classes)

        if transformer_pretrain is not None:
            self.transformer_encoder.load_state_dict(torch.load(transformer_pretrain, map_location='cpu'))

    def forward(self, x):
        # input shape: (B, T, C, H, W))
        
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.backbone(x)          # (B, 512, T)
        x = self.dropout_backbone(x)
        
        # Transformer Encoder a (T, B, C) = (seq, batch, features) formátumban várja a bemenetet
        x = x.permute(2, 0, 1)  # (T, B, 512)
        x = self.transformer_encoder(x)  # (T, B, 512)
        
        # Globális időbeli pool (átlagolás T dimenzió mentén)
        x = x.mean(dim=0)  # (B, 512)
        
        # Klasszifikációs fej
        x = self.dropout(x)
        logits = self.classifier(x)  # (B, num_classes)
        return logits

if __name__ == '__main__':
    model = GestureRecognizer()
    input_tensor = torch.randn(2, 32, 3, 112, 112)
    with torch.no_grad():
        output = model(input_tensor)
    print(output.shape)  # torch.Size([2, 27])
    print(output)
    probs, class_idx = torch.max(output, dim=1)
    for i in range(output.size(0)):
        print(f"A {i+1}. minta legvalószínűbb osztálya: {class_idx[i].item()}, valószínűsége: {probs[i].item():.4f}")
