from ssl_components.utils import *

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 image_size, 
                 patch_size, 
                 in_channels=3, 
                 embed_dim = 768 ):
        
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2

        self.proj = nn.Conv2d(in_channels=in_channels,
                                out_channels=embed_dim,
                                kernel_size=self.patch_size,
                                stride=self.patch_size)
        
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.permute(0,2,1)
        return x

class MLP(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim_ratio = 4,
                 dropout = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(in_features=dim, 
                                out_features=dim*hidden_dim_ratio)
        self.linear2 = nn.Linear(in_features=dim*hidden_dim_ratio,
                                 out_features=dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
    
    def forward(self, x):
        net = nn.Sequential(self.linear1, 
                            self.gelu, 
                            self.dropout,
                            self.linear2, 
                            self.dropout)

        return net(x)
    
class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 dropout = 0.0
                 ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        self.kqv_proj = nn.Linear(dim, dim*3)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.scale = (self.head_dim)**(-0.5)

    def forward(self, x):
        B, N, D = x.shape
        kqv = self.kqv_proj(x)
        kqv = kqv.reshape(B, N, 3, self.num_heads, self.head_dim)

        ## kqv: 3, B, num_heads, N, head_dim

        kqv = kqv.permute(2,0,3,1,4)

        ## k,q,v: B, num_heads, N, head_dim

        k,q,v = kqv

        ## attn_score : B, num_heads, N, N ---> NxN Matrix

        attn_score = (q@k.transpose(-2,-1))*self.scale

        ## Normalize attention scores across each row ---> dim = 2

        attn_score = attn_score.softmax(-1)
        attn_score = self.dropout(attn_score)
        x = attn_score@v

        ## x shape: B, N, num_heads, head_dim

        x = x.permute(0,2,1,3)

        ## combine the attention heads

        x = x.reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads,
                 hidden_dim_ratio = 4,
                 dropout = 0.0):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.mlp = MLP(dim, hidden_dim_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self,
                 image_size, 
                 patch_size, 
                 in_channels=3, 
                 embed_dim = 768,
                 depth = 3,
                 num_heads = 12,
                 hidden_dim_ratio = 4,
                 dropout = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, 
                                          patch_size,
                                          in_channels,
                                          embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            [
                Transformer(embed_dim,
                            num_heads,
                            hidden_dim_ratio,
                            dropout)
                
                for _ in range(depth)
            ]
        )
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        h = self.projector(x)

        return x, h

class LensModel(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 in_channels=3, 
                 embed_dim = 768,
                 depth = 3,
                 num_heads = 12,
                 hidden_dim_ratio = 4,
                 dropout = 0.0
                 ):
        super().__init__()
        self.vit_encoder = ViTEncoder(image_size,
                                      patch_size,
                                      in_channels,
                                      embed_dim,
                                      depth,
                                      num_heads,
                                      hidden_dim_ratio,
                                      dropout)

        def _make_decoder(final_activation=None):
            layers = [
                nn.ConvTranspose2d(embed_dim, 256, patch_size, patch_size),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 1, 3, padding=1),
            ]
            if final_activation is not None:
                layers.append(final_activation)
            return nn.Sequential(*layers)

        self.convergence_decoder = _make_decoder()
        self.potential_decoder = _make_decoder()
        # Sigmoid because the preprocessed image is in [0, 1].
        self.image_decoder = _make_decoder(final_activation=nn.Sigmoid())
        # Raw flux spans ~1e-8 to ~20; no Sigmoid. Target is log + z-normalised.
        self.raw_image_decoder = _make_decoder()

        self.num_patch_row = image_size // patch_size

    def forward(self, x):
        vit_encoding, latent_encoding = self.vit_encoder(x)

        # With 64x64 inputs and 16x16 patches: (B, 16, 768).
        B, N, D = vit_encoding.shape
        H_patch = W_patch = self.num_patch_row  # 4x4 patch grid for the default input.

        # For the default input, feature_map has shape (B, 768, 4, 4).
        feature_map = vit_encoding.permute(0, 2, 1).reshape(B, D, H_patch, W_patch)
        latent_encoding = latent_encoding.permute(0, 2, 1)  # (B, 768, 16) for the default input.

        convergence = self.convergence_decoder(feature_map)
        potential = self.potential_decoder(feature_map)
        image_recon = self.image_decoder(feature_map)
        raw_image_recon = self.raw_image_decoder(feature_map)

        return feature_map, latent_encoding, potential, convergence, image_recon, raw_image_recon

        




        
