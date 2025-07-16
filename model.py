# model.py (已更新，可返回特征)
import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

class DGCNN(nn.Module):
    def __init__(self, num_classes, k=20):
        super().__init__()
        self.k = k

        mlp1 = nn.Sequential(nn.Linear(6, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        self.conv1 = DynamicEdgeConv(nn=mlp1, k=self.k)
        mlp2 = nn.Sequential(nn.Linear(128, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        self.conv2 = DynamicEdgeConv(nn=mlp2, k=self.k)
        mlp3 = nn.Sequential(nn.Linear(128, 128, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.conv3 = DynamicEdgeConv(nn=mlp3, k=self.k)
        self.conv_final = nn.Sequential(nn.Conv1d(256, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, data, return_features=False):
        pos, batch = data.pos, data.batch
        
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        
        x_combined = torch.cat([x1, x2, x3], dim=1)
        x_final = self.conv_final(x_combined.unsqueeze(-1))
        global_feature = global_max_pool(x_final.squeeze(-1), batch)
        
        logits = self.classifier(global_feature)
        
        if return_features:
            return logits, global_feature
        else:
            return logits