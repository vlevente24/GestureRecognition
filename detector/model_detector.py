import torch
import torch.nn as nn
import torchvision.models as models


class BasicBlock3D(nn.Module):
    """3D ResNet BasicBlock - similar structure as 2D BasicBlock"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # First 3D convolution
        # stride=(1, stride, stride): T dimension remains unchanged, only spatial downsampling
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),  # Temporal stride=1 ALWAYS!
            padding=(1, 1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second 3D convolution
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),  # Always stride=1
            padding=(1, 1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out
    

def make_layer( in_channels, out_channels, blocks, stride=1):
        """ResNet layer creation (2 BasicBlock)"""
        downsample = None
        
        # If the input and output dimensions differ, we need a downsample layer
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(1, stride, stride),  # Temporal stride=1!
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        
        # First block: transition (dimensions may change)
        layers.append(BasicBlock3D(in_channels, out_channels, stride, downsample))
        
        # Other blocks: refinement (same dimensions)
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)


# Same as in detector 2, but with different parameters


class GestureDetectorConv3d(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # 2D ResNet18 backbone (az utolsó classification réteg nélkül)
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.extralayer = make_layer(128, 256, blocks=2, stride=2)
        
        # Transformer Encoder blokk, amely tartalmaz Multihead Attention-t is
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Klasszifikációs fej
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # 2D backbone: batch és time egyesítése, azaz (B*T, C, H, W)
        x = x.view(B*T, C, H, W)
        features = self.backbone(x)  # (B*T, 512, 1, 1)
        
        _, c, h, w = features.shape
        features = features.view(B, T, c, h, w)
        features = features.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        features = self.extralayer(features)

        features = features.mean(dim=[3, 4])  # (B, C, T)
        features = features.permute(2, 0, 1)  # (T, B, C)

        x = self.transformer_encoder(features)  # (T, B, 512)

        # Globális időbeli pool (átlagolás T dimenzió mentén)
        x = x.mean(dim=0)  # (B, 512)
        
        # Klasszifikációs fej
        x = self.dropout(x)
        logits = self.classifier(x)  # (B, num_classes)
        return logits
    