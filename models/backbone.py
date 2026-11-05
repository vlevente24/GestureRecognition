import torch
import torch.nn as nn


class Squeeze(nn.Module):
    """Custom layer to remove dimensions of size 1"""
    
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)


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


# ===== 3D ResNet18 model =====


class ResNet3D18(nn.Module):
    """
    3D ResNet18 - similar to 2D ResNet18 but with 3D convolutions
    
    Parameters
    ----------
    num_classes: int
        Number of output classes for classification
    input_channels: int
        Number of input channels (e.g., 3 for RGB)
    classify: bool
        If True, includes the final classification layer. If False, outputs features before classification.

    Features
    --------
    x: 3D feature tensor
            (B, C, T, H, W)

    Returns
    -------
        output: (B, num_classes)
            if classify=False: (B, 512, T)
    """

    def __init__(self, num_classes=27, input_channels=3, pretrained=None, classify=True):
        super().__init__()

        self.classify = classify
        
        # Stem (starting layer): 7x7 conv + maxpool
        # Temporal stride=1 -> T does not change
        layer0_conv1 = nn.Conv3d(
            input_channels, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),       # Temporal stride=1!
            padding=(1, 3, 3),  
            bias=False
        )
        layer0_bn1 = nn.BatchNorm3d(64)
        layer0_relu = nn.ReLU(inplace=True)
        layer0_maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),  # Temporal kernel=1
            stride=(1, 2, 2),       # Temporal stride=1!
            padding=(0, 1, 1)
        )

        # Stem layer
        self.layer0 = nn.Sequential(
            layer0_conv1,
            layer0_bn1,
            layer0_relu,
            layer0_maxpool
        )
        
        # ResNet layers (same as ResNet18, but 3D)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)    # 64 channels
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)   # 128 channels
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)  # 256 channels
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)  # 512 channels
        
        # Global Average Pooling
        # Spatial pooling, keep T dimension
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # (T, 1, 1)
            Squeeze(dim=[-1, -2])  # Remove last two dims
        )
        
        # Temporal pooling (along T dimension)
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # T → 1
            Squeeze(dim=-1)  # Remove last dim
        )
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)
        
        # Init
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'))
        else:
            self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
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
    
    def _initialize_weights(self):
        """Initialize weights (He initialization)"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Input: (B, C, T, H, W)"""
        
        # Stem
        x = self.layer0(x)     # (B, 64, T, h, w)
        
        # ResNet layers
        x = self.layer1(x)     # (B, 64, T, h, w)
        x = self.layer2(x)     # (B, 128, T, h', w')
        x = self.layer3(x)     # (B, 256, T, h", w")
        x = self.layer4(x)     # (B, 512, T, h"', w"')
        
        # Spatial pooling
        x = self.avgpool(x)    # (B, 512, T)

        if not self.classify:
            return x  # Return features (B, 512, T) before temporal pooling and classification
        
        # Temporal pooling
        x = self.temporal_pool(x)
        
        # Classifier
        x = self.fc(x)  # (B, num_classes)
        
        return x
