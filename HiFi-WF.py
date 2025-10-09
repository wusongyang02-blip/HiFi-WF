import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class HierarchicalFeatureInteraction(nn.Module):
    """Hierarchical Feature Interaction module using cross-attention mechanism
    to model dependencies between main and sub features"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim  # Dimension of input features
        # Multi-head cross attention module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,  # Dimension of embeddings
            num_heads=8,  # Number of attention heads
            batch_first=True,  # Batch dimension first
            dropout=0.1  # Dropout rate
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)  # Layer normalization after attention
        self.norm2 = nn.LayerNorm(feature_dim)  # Layer normalization after FFN
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Learnable fusion coefficients
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Weight for main feature residual connection
        self.beta = nn.Parameter(torch.tensor(0.5))   # Weight for sub feature residual connection

    def forward(self, main_feat, sub_feat):
        """
        Args:
            main_feat: Main branch features (B, D)
            sub_feat: Sub branch features (B, D)
        Returns:
            Enhanced main and sub features after interaction
        """
        combined = torch.stack([main_feat, sub_feat], dim=1)  # (B, 2, D)
        attn_out, _ = self.cross_attention(combined, combined, combined)  # Self-attention
        attn_out = self.norm1(combined + attn_out)  # Residual + norm
        ffn_out = self.ffn(attn_out)  # Feed-forward
        enhanced = self.norm2(attn_out + ffn_out)  # Residual + norm
        
        # Feature fusion with residual connections
        enhanced_main = self.alpha * main_feat + (1 - self.alpha) * enhanced[:, 0]
        enhanced_sub = self.beta * sub_feat + (1 - self.beta) * enhanced[:, 1]
        
        return enhanced_main, enhanced_sub

class HiFi_WF(nn.Module):    
    """Hierarchical Feature Interaction Network for Waveform classification"""
    def __init__(self, num_main=50, num_sub=500, in_channels=2, base_channels=64):
        super().__init__()
        self.num_main = num_main  # Number of main categories
        self.num_sub = num_sub    # Number of sub categories
        self.per_main_sub = num_sub // num_main  # Sub categories per main category
        
        # 1. Feature Extractor: Convolutional backbone for feature extraction
        self.scanner = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, padding=3, stride=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),
            
            self._make_conv_block(base_channels, base_channels*2),
            self._make_conv_block(base_channels*2, base_channels*4),
            self._make_conv_block(base_channels*4, base_channels*8),
            
            nn.Conv1d(base_channels*8, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)  # Fixed-length feature mapping
        )
        
    # 2. Feature Enhancer: Attention mechanism for feature refinement
        self.feature_enhancer = nn.Sequential(
            ChannelAttention(256, reduction=16),  # Channel-wise attention
            SpatialAttention(kernel_size=5),      # Spatial-wise attention
            nn.ReLU()
        )
        
    # 3. Dual-Path Segmentation: Feature splitting mechanism
        self.dual_path_split = nn.Sequential(
            nn.Conv1d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 2, 1),  # Generate 2 attention maps
            nn.Softmax(dim=1)
        )
        
    # 4. Hierarchical Feature Encoder: Shared encoder for feature projection
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(256, 192, 3, padding=1, groups=8),  # Depth-wise convolution
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)  # Fixed-length encoding
        )
        
    # 5. Hierarchical Feature Interaction module
        self.feature_interaction = HierarchicalFeatureInteraction(feature_dim=192*16)
        
    # 6. Classifiers: Dual-path main and sub category predictors
        self.main_head1 = self._create_main_head(192*16, num_main)
        self.sub_head1 = self._create_sub_head(192*16, num_main, self.per_main_sub)
        self.main_head2 = self._create_main_head(192*16, num_main)
        self.sub_head2 = self._create_sub_head(192*16, num_main, self.per_main_sub)
        
        self.apply(self._init_weights)  # Initialize weights
        
    def _make_conv_block(self, in_c, out_c):
        """Create a convolutional block with Conv-BN-ReLU-Pooling"""
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True)
        )
    
    def _create_main_head(self, in_features, num_classes):
        """Create classifier head for main categories"""
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _create_sub_head(self, in_features, num_main, per_main_sub):
        """Create enhanced classifier head for sub categories"""
        return EnhancedSubHead(
            num_main=num_main,
            per_main_sub=per_main_sub,
            in_features=in_features,
            hidden_size=256
        )
    
    def _init_weights(self, m):
        """Initialize layer weights"""
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input waveform features (B, C, T)
        Returns:
            main_outputs: Main category predictions (B, 2, num_main)
            sub_outputs: Sub category predictions (B, 2, num_sub)
        """
        x = self.scanner(x)  # Feature extraction
        x = self.feature_enhancer(x)  # Feature enhancement
        tab_att = self.dual_path_split(x)  # Path attention maps
        
        # Dual-path feature processing
        weighted1 = x * tab_att[:, 0:1]  # First path weighting
        encoded1 = self.shared_encoder(weighted1)
        flat1 = encoded1.view(encoded1.size(0), -1)  # Flatten
        
        weighted2 = x * tab_att[:, 1:2]  # Second path weighting
        encoded2 = self.shared_encoder(weighted2)
        flat2 = encoded2.view(encoded2.size(0), -1)  # Flatten
        
        flat1, flat2 = self.feature_interaction(flat1, flat2)  # Feature interaction
        
        # Predictions from dual paths
        main_out1 = self.main_head1(flat1)
        sub_out1 = self.sub_head1(flat1, main_out1)
        
        main_out2 = self.main_head2(flat2)
        sub_out2 = self.sub_head2(flat2, main_out2)
        
        # Aggregate outputs
        main_outputs = torch.stack([main_out1, main_out2], dim=1)
        sub_outputs = torch.stack([sub_out1, sub_out2], dim=1)
        
        return main_outputs, sub_outputs

class ChannelAttention(nn.Module):
    """Squeeze-excitation channel attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # Reduction
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),  # Expansion
            nn.Sigmoid()  # Channel attention weights
        )
    
    def forward(self, x):
        b, c, _ = x.shape
        avg_out = self.avg_pool(x).view(b, c)  # (B, C)
        channel_att = self.fc(avg_out)  # (B, C)
        return x * channel_att.view(b, c, 1)  # (B, C, T) with channel weights

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for 1D features"""
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2)  # Spatial convolution
        self.sigmoid = nn.Sigmoid()  # Spatial attention weights
    
    def forward(self, x):
        spatial_att = torch.mean(x, dim=1, keepdim=True)  # (B, 1, T)
        spatial_att = self.sigmoid(self.conv(spatial_att))  # Spatial weighting
        return x * spatial_att  # (B, C, T) with spatial weights

class EnhancedSubHead(nn.Module):
    """Enhanced sub-category classifier with parent category information injection"""
    def __init__(self, num_main, per_main_sub, in_features, hidden_size):
        super().__init__()
        self.num_main = num_main  # Number of main categories
        self.per_main_sub = per_main_sub  # Sub categories per main category
        
        # Parent category information injection module
        self.parent_info_injection = nn.Sequential(
            nn.Linear(num_main, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Feature transformation module
        self.transform = nn.Sequential(
            nn.Linear(in_features + hidden_size // 4, hidden_size * 2), 
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Dynamic constraint mask generator
        self.constraint_gen = nn.Sequential(
            nn.Linear(num_main, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_main * per_main_sub)
        )
        
        # Multi-scale classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_main * per_main_sub)
            for _ in range(3)
        ])
        
    def forward(self, features, main_pred):
        """
        Args:
            features: Input features (B, D)
            main_pred: Main category predictions (B, num_main)
        Returns:
            sub_logits: Sub category logits (B, num_sub)
        """
        parent_info = self.parent_info_injection(main_pred)  # (B, hidden_size//4)
        
        # Feature fusion
        enriched_features = torch.cat([features, parent_info], dim=1)
        
        # Feature transformation
        trans_feat = self.transform(enriched_features)
        
        # Generate dynamic constraint mask
        constraint_mask = torch.sigmoid(self.constraint_gen(main_pred))
        
        # Multi-classifier fusion
        sub_logits = torch.stack([classifier(trans_feat) for classifier in self.classifiers])
        sub_logits = torch.mean(sub_logits, dim=0)  # Average across classifiers
        
        return sub_logits * constraint_mask  # Apply dynamic constraints