import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LearnableAudio(nn.Module):
    """Base class for audio-oriented mask generation.
    height: frequency dimension
    width: time dimension
    num_channels: number of output channels for the mask (ex: 1 for mono, or more if needed)
    """
    def __init__(self, height: int, width: int, num_channels: int = 1):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward() method.")

class AudioMaskGeneratorCNN(LearnableAudio):
    """Lightweight 2D CNN으로 time-frequency 인접 정보를 반영하는 mask generator."""
    def __init__(self,
                 height: int,
                 width: int,
                 num_channels: int = 1,
                 hidden_dim: int = 32):
        super().__init__(height, width, num_channels)
        
        # 소규모 CNN 모델 정의
        # kernel_size=3으로 주변 bin 정보를 가볍게 수용
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            
            nn.Conv2d(hidden_dim, self.num_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 초기 입력으로 사용할 learnable parameter(1채널짜리 2D 텐서)
        # 또는 uv_grid를 넣어도 되지만 여기선 단순히 random tensor로 시작
        self.init_param = nn.Parameter(torch.randn(1, 1, height, width))

    def forward(self):
        """
        Returns:
            mask: shape [1, num_channels, height, width]
        """
        x = self.init_param  # [1, 1, H, W]
        mask = self.model(x) # [1, num_channels, H, W]
        return mask.squeeze(0)


class AudioMaskGeneratorTransformer(LearnableAudio):
    """Small Transformer Encoder로 time-frequency bin 간의 global context를 반영하는 mask generator."""
    def __init__(self,
                 height: int,
                 width: int,
                 num_channels: int = 1,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 num_layers: int = 2):
        """
        embed_dim: Transformer 내부 임베딩 차원
        num_heads: Multi-head attention 개수
        num_layers: Transformer Encoder Layer 수
        """
        super().__init__(height, width, num_channels)
        
        self.height = height
        self.width = width
        self.num_patches = height * width  # time-frequency bin 개수
        
        # learnable input parameter (flatten 후 Transformer로 처리)
        self.init_param = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # 간단한 Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim*2,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 최종 output -> mask 값으로 매핑하는 small projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, num_channels),
            nn.Sigmoid()
        )

    def forward(self):
        """
        Returns:
            mask: shape [1, num_channels, height, width]
        """
        # [1, num_patches, embed_dim]
        x = self.init_param
        
        # Self-Attention으로 global context 반영
        x = self.transformer(x)  # [1, num_patches, embed_dim]
        
        # 최종 projection -> [1, num_patches, num_channels]
        x = self.proj(x)

        # 2D로 reshape -> [1, num_channels, height, width]
        mask = x.view(1, self.num_channels, self.height, self.width)
        return mask.squeeze(0)
