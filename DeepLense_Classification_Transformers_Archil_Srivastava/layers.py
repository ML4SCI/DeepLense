import torch
from torch.nn import Linear, Dropout, ReLU, Embedding, GELU, LayerNorm, MultiheadAttention, Sequential
import torch.nn.functional as F

class FFN(torch.nn.Module):
    '''
    Custom Feedforward Network with dropout after every layer
    '''
    def __init__(self, input_units, hidden_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        layers = []
        for units in hidden_units:
            layers.append(Linear(input_units, units))
            layers.append(GELU())
            layers.append(Dropout(p=dropout_rate))
            input_units = units
        self.ffn = Sequential(*layers)
  
    def forward(self, x):
        return self.ffn(x)

class Patches(torch.nn.Module):
    '''
    Code picked from https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/8
    '''
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(patches.shape[0], patches.shape[1] * patches.shape[2], -1)
        return patches


class PatchEncoder(torch.nn.Module):
    def __init__(self, num_patches, patch_dim, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Linear(patch_dim, projection_dim)
        self.position_embedding = Embedding(num_patches, projection_dim)
    
    def forward(self, patch):
        positions = torch.arange(0, self.num_patches, dtype=torch.long).to(next(self.parameters()).device)
        # Add patch embeddings and position embeddings
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class Transformer(torch.nn.Module):
    '''
    Transformer Block
    Input shape: (batch_size, num_patches, projection_dim)
    Output shape: (batch_size, num_patches, projection_dim)
    '''
    def __init__(self, num_patches, transformer_units, num_heads, projection_dim,
                 epsilon=1e-6, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        input_shape = (num_patches, projection_dim)
        self.norm1 = LayerNorm(input_shape, eps=epsilon)
        self.attention = MultiheadAttention(projection_dim, num_heads=num_heads, dropout=dropout_rate)
        self.norm2 = LayerNorm(input_shape, eps=epsilon)
        self.ffn = FFN(input_shape[-1], transformer_units, dropout_rate)
  
    def forward(self, encoded_patch):
        x1 = self.norm1(encoded_patch)
        attention_output = self.attention(x1, x1, x1, need_weights=False)[0]
        x2 = attention_output + encoded_patch # Add()([attention_output, encoded_patch])
        x3 = self.norm2(x2)
        x3 = self.ffn(x3)
        output = x3 + x2 # Add()([x3, x2])
        return output