import torch
import torch.nn as nn
import torchvision.models as models

class GestureRecognizer(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

         # 2D ResNet18 backbone (az utolsó classification réteg nélkül)
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512  # ResNet utolsó conv réteg csatornaszáma
        
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
    

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # 2D backbone: batch és time egyesítése, azaz (B*T, C, H, W)
        x = x.view(B*T, C, H, W)
        features = self.backbone(x)  # (B*T, 512, 1, 1)
        features.squeeze_(-1).squeeze_(-1)  # (B*T, 512)
        features = features.view(B, T, self.feature_dim)  # (B, T, 512)

        # Transformer Encoder a (T, B, C) = (seq, batch, features) formátumban várja a bemenetet
        features = features.permute(1, 0, 2)  # (T, B, 512) a Transformer-hez
        x = self.transformer_encoder(features)  # (T, B, 512)

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
