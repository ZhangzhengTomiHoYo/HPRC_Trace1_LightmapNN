import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs = torch.pow(2.0, torch.arange(num_freqs)).float()

    def forward(self, x):
        # x: [N, D]
        if self.num_freqs == 0:
            return x
        
        freqs = self.freqs.to(x.device)
        # [N, D, 1] * [1, 1, num_freqs] -> [N, D, num_freqs]
        args = x.unsqueeze(-1) * freqs.view(1, 1, -1) * np.pi
        args = args.reshape(x.shape[0], -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ExampleModel(nn.Module):
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=256):
        super(ExampleModel, self).__init__()
        
        # --- Config for optimization ---
        self.grid_resolutions = [16, 32, 64, 128]  # Multi-scale grid set
        self.feat_dim = 2             # Channels per grid
        self.pe_freqs = 6             # Positional encoding frequencies
        
        # 1. Multi-scale Feature Grids
        self.grids = nn.ModuleList()
        for res in self.grid_resolutions:
            # Init grid: [1, feat_dim, res, res]
            grid = nn.Parameter(torch.zeros(1, self.feat_dim, res, res, dtype=torch.float32))
            # Standard initialization for grid features
            nn.init.uniform_(grid, -0.01, 0.01)
            self.grids.append(grid)
            
        # 2. Positional Encoding
        # We encoded x, y, t inputs. Input x to forward is (u, v, t) -> 3 dims
        self.pe = PositionalEncoding(self.pe_freqs)
        
        # Calculate MLP input dimension
        # Raw coords (3) + PE features (3 * 2 * freqs) + Grid features (num_grids * feat_dim)
        mlp_in_dim = 3 + (3 * 2 * self.pe_freqs) + (len(self.grid_resolutions) * self.feat_dim)
        
        # 3. MLP Network
        self.model = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        # x shape: [N, 3] usually (u, v, t)
        
        # Part A: Grid Sampling (Spatial only: u, v)
        # Normalize coords from [0, 1] to [-1, 1] for grid_sample
        # x[:, 0] is u (y-axis in code logic?), x[:, 1] is v
        pixel_coords = x[:, [0, 1]].unsqueeze(0).unsqueeze(2) * 2.0 - 1.0 # [1, N, 1, 2]
        
        grid_features = []
        for grid in self.grids:
            # Sample from grid. bilinear interpolation.
            # align_corners=True matches the -1..1 range to corner pixels exactly
            feat = F.grid_sample(grid, pixel_coords, mode='bilinear', padding_mode='border', align_corners=True)
            # feat: [1, C, N, 1] -> [N, C]
            feat = feat.squeeze(0).squeeze(-1).permute(1, 0)
            grid_features.append(feat)
            
        grid_features = torch.cat(grid_features, dim=-1)
        
        # Part B: Positional Encoding (Spatial + Temporal)
        pe_features = self.pe(x)
        
        # Part C: Concatenate and Predict
        # Concat: [Raw Input, PE, Grid Features]
        features_in = torch.cat([x, pe_features, grid_features], dim=-1)
        
        return self.model(features_in)