import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer that allows landmarks to attend to image or text features
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # Key/Value projections for the other modality (image or text)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, landmark_features, context_features, attention_mask=None):
        """
        Args:
            landmark_features (torch.Tensor): [batch_size, num_landmarks, hidden_dim]
            context_features (torch.Tensor): [batch_size, context_length, hidden_dim]
            attention_mask (torch.Tensor, optional): [batch_size, num_landmarks, context_length]
                1 = attend, 0 = ignore
        """
        batch_size = landmark_features.shape[0]
        
        # Project queries from landmark features
        q = self.q_proj(landmark_features)
        # Project keys/values from context features (image or text)
        k = self.k_proj(context_features)
        v = self.v_proj(context_features)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None:
            # Expand mask for multiple heads
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            # Apply mask by setting masked positions to -inf
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights
    

class LandmarkTrajectoryProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Like PaliGemma, project to the language model's hidden dimension
        self.projection_dim = config.text_config.hidden_size
        
        
        # For landmark trajectory shape (batch, 10, 478, 3)
        self.input_proj = nn.Linear(478 * 3, 512)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                activation='gelu',
                batch_first=True,
                dropout=0.1
            ),
            num_layers=6
        )
        
        self.proj = nn.Linear(512, self.projection_dim)
        
        # Cross-attention to image features
        self.cross_attn_image = CrossAttentionLayer(
            hidden_dim=self.projection_dim,
            num_heads=8,
            dropout=0.05
        )
        
        # Cross-attention to text features
        self.cross_attn_text = CrossAttentionLayer(
            hidden_dim=self.projection_dim,
            num_heads=8,
            dropout=0.05
        )
        
        # Final layer norm and feedforward
        self.layer_norm1 = nn.LayerNorm(self.projection_dim)
        self.layer_norm2 = nn.LayerNorm(self.projection_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.projection_dim * 4, self.projection_dim)
        )
        
    def forward(self, landmark_values, image_features=None, text_features=None, 
                image_mask=None, text_mask=None):
        
        if landmark_values is None:
            return None
        
        b, t, l, c = landmark_values.size()
        x = landmark_values.view(b, t, l*c)  # shape: (b, 10, 478*3)
        x = self.input_proj(x)  # shape: (b, 10, 512)
        x = self.transformer(x)  # shape: (b, 10, 512)
        x = self.proj(x)  # shape: (b, 10, hidden_size)
        
        
        landmark_features = x
        
        # Cross-attention with image features if available
        if image_features is not None:
            image_context, _ = self.cross_attn_image(landmark_features, image_features, image_mask)
            # Add & Norm
            x = self.layer_norm1(x + image_context)
        
        # Cross-attention with text features if available
        if text_features is not None:
            text_context, _ = self.cross_attn_text(x, text_features, text_mask)
            # Add & Norm
            x = self.layer_norm2(x + text_context)
        
        ff_output = self.feedforward(x)
        x = x + ff_output  # Residual connection
        return x