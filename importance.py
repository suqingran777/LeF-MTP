import torch
import numpy as np

def calculate_geometric_importance(point_cloud: torch.Tensor) -> float:
    """
    计算一个点云的几何重要性分数。
    我们使用点到质心距离的标准差作为复杂度的代理指标。

    Args:
        point_cloud (torch.Tensor): 一个形状为 [N, 3] 的点云张量，
                                    其中 N 是点的数量。
                                    假设该点云已经过归一化处理。

    Returns:
        float: 代表该点云几何复杂度的单一分数。
    """
    if not isinstance(point_cloud, torch.Tensor):
        raise TypeError("输入必须是一个PyTorch张量。")
    if point_cloud.dim() != 2 or point_cloud.shape[1] != 3:
        raise ValueError("输入张量的形状必须是 [N, 3]。")
    if point_cloud.shape[0] == 0:
        return 0.0

    # 1. 计算点云的质心 (centroid)
    centroid = torch.mean(point_cloud, dim=0)  # 形状: [3]

    # 2. 计算每个点到质心的欧几里得距离
    #    point_cloud - centroid 会进行广播操作
    distances = torch.linalg.norm(point_cloud - centroid, dim=1)  # 形状: [N]

    # 3. 计算这些距离的标准差
    importance_score = torch.std(distances)

    return importance_score.item()


# --- 主程序：演示和验证该函数的功能 ---
if __name__ == '__main__':
    print("--- 几何重要性计算函数功能验证 ---")

    # 创建两个模拟的点云来进行对比
    # 假设它们都已被归一化到单位球内

    # 1. 创建一个简单的形状：球体上的点
    #    这些点到中心的距离应该非常接近，所以标准差会很小
    phi = np.pi * (np.sqrt(5.) - 1.)  # 黄金角
    n_points = 1024
    i = np.arange(n_points)
    y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)  # radius at y
    theta = phi * i  # golden angle increment
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    sphere_points = torch.from_numpy(np.stack([x, y, z], axis=-1).astype(np.float32))

    # 2. 创建一个复杂的形状：在单位球内随机分布，但半径不均匀
    #    这将导致点到中心的距离差异很大，标准差会很大
    random_radii = torch.rand(n_points)
    complex_points = sphere_points * random_radii.unsqueeze(1) # 用不同的半径拉伸每个点

    # 计算两种形状的重要性分数
    simple_shape_score = calculate_geometric_importance(sphere_points)
    complex_shape_score = calculate_geometric_importance(complex_points)

    print(f"\n简单形状（球体）的重要性分数: {simple_shape_score:.4f}")
    print(f"复杂形状（不规则体）的重要性分数: {complex_shape_score:.4f}")

    # 断言：复杂形状的分数应该远大于简单形状的分数
    assert complex_shape_score > simple_shape_score
    print("\n✅ 验证成功：复杂形状的得分显著高于简单形状。")

    # 更高级的方法说明
    print("\n注：这是一种高效的复杂度代理方法。更高级的方法可以引入表面曲率（surface curvature）的计算，但这通常需要更复杂的库（如Open3D）和更多的计算量。")