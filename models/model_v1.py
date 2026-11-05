import torch
import torch.nn as nn
import torchvision.models as models

class GestureRecognizer(nn.Module):
    def __init__(self, num_classes=27, sample_duration=32):
        super().__init__()
        self.sample_duration = sample_duration

        # 2D ResNet18 backbone (az utolsó classification réteg nélkül)
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 512  # ResNet utolsó conv réteg csatornaszáma

        # 3D convolution a spatio-temporális jellemzők kinyerésére
        self.conv3d = nn.Conv3d(
            in_channels=self.feature_dim,
            out_channels=512,
            kernel_size=(3,3,3),
            stride=(1,1,1),
            padding=(1,1,1)
        )
        self.ln3d = nn.LayerNorm(512) # layer normalization
        self.relu = nn.ReLU(inplace=True)
        self.dropout3d = nn.Dropout3d(p=0.1)
        
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
        features = self.backbone(x)  # (B*T, 512, h', w')

        # Visszaalakítás 5D tensorra (B, feature_dim, T, h', w')
        features = features.view(B, T, self.feature_dim, features.size(-2), features.size(-1))
        features = features.permute(0, 2, 1, 3, 4)  # (B, C, T, h', w')
        
        # 3D conv blokk
        x = self.conv3d(features)  # (B, 512, T, h', w')
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, h', w', 512) a LayerNorm-hoz
        x = self.ln3d(x)  # layer normalization csatorák mentén
        x = x.permute(0, 4, 1, 2, 3)  # vissza (B, 512, T, h', w')
        x = self.relu(x)
        x = self.dropout3d(x)

        # Spatial pooling = térbeli dimenziók átlaga -> (B, 512, T)
        x = x.mean(dim=[3,4])  # átlagolás h' és w' dimenziókon
        
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
