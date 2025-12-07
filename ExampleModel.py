import torch
import torch.nn as nn
import torch.nn.functional as F

class ExampleModel(nn.Module):
    def __init__(self, input_dim = 6, output_dim = 3, hidden_dim = 256):
        super(ExampleModel, self).__init__()
        
        self.featuremap = torch.nn.Parameter(torch.empty((1, 3, 32, 32), dtype=torch.float32, requires_grad=True))
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
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
        with torch.no_grad():
            nn.init.normal_(self.featuremap, mean=0.0, std=0.1)
    
    def forward(self, x):
        feature = F.grid_sample(self.featuremap, x[:, [0, 1]].unsqueeze(0).unsqueeze(2) * 2.0 - 1.0, mode='bilinear', padding_mode='border', align_corners=False)
        feature = feature.squeeze(0).squeeze(-1).permute(1, 0)
        return self.model(torch.cat([x, feature], dim=-1))