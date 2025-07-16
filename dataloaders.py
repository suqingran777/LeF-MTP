# dataloaders.py (已修正对点云的预处理)
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet, ShapeNet, ScanObjectNN
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# 新增的、为点云设计的自定义转换类
class FixedPoints(object):
    """
    一个自定义的转换，用于将任何点云采样或上采样到固定的点数。
    """
    def __init__(self, num):
        self.num = num

    def __call__(self, data: Data) -> Data:
        pos = data.pos
        num_points = pos.size(0)

        if num_points == self.num:
            # 如果点数正好，直接返回
            return data
        
        if num_points > self.num:
            # 如果点数过多，随机选择不重复的索引
            choice = torch.randperm(num_points)[:self.num]
            data.pos = pos[choice]
        else:
            # 如果点数不足，随机选择可重复的索引进行上采样
            choice = torch.randint(0, num_points, (self.num - num_points,))
            # 将重复采样的点与原始点拼接起来
            duplicate_points = pos[choice]
            data.pos = torch.cat([pos, duplicate_points], dim=0)
            
        return data

# (get_modelnet40_dataloaders 和 get_shapenet_dataloaders 函数保持不变，此处省略)
def get_modelnet40_dataloaders(batch_size=32, sample_points=1024, num_workers=4):
    path = 'data/ModelNet40'
    pre_transform = T.Compose([
        FixedPoints(sample_points), # ModelNet40也是点云，同样可以使用这个方法
        T.NormalizeScale(),
    ])
    train_dataset = ModelNet(root=path, name='40', train=True, pre_transform=pre_transform)
    test_dataset = ModelNet(root=path, name='40', train=False, pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_shapenet_dataloaders(batch_size=32, sample_points=1024, num_workers=4, category='Chair'):
    path = f'data/ShapeNet_{category}'
    pre_transform = T.Compose([
        FixedPoints(sample_points), # ShapeNet也是点云，同样可以使用
        T.NormalizeScale(),
    ])
    train_dataset = ShapeNet(root=path, categories=[category], split='trainval', pre_transform=pre_transform)
    test_dataset = ShapeNet(root=path, categories=[category], split='test', pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
# ---

def get_scanobjectnn_dataloaders(batch_size=32, sample_points=1024, num_workers=4):
    """
    加载并预处理ScanObjectNN数据集。
    """
    path = 'data/ScanObjectNN'
    
    # ***************************************************************
    # *** 核心修正部分 START                     ***
    # ***************************************************************
    # 我们用自己编写的FixedPoints替换掉不兼容的T.SamplePoints
    pre_transform = T.Compose([
        FixedPoints(sample_points),
        T.NormalizeScale(), # 归一化依然是个好习惯
    ])
    # ***************************************************************
    # *** 核心修正部分 END                      ***
    # ***************************************************************
    
    print("正在加载ScanObjectNN数据集...")
    train_dataset = ScanObjectNN(root=path, name='main_split', split='training', pre_transform=pre_transform)
    test_dataset = ScanObjectNN(root=path, name='main_split', split='test', pre_transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("\nScanObjectNN数据加载完成！")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别数: {train_dataset.num_classes}")
    
    return train_loader, test_loader