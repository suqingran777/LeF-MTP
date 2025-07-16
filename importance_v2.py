import torch
import numpy as np
import open3d as o3d

def calculate_curvature_importance(point_cloud_np: np.ndarray) -> float:
    """
    使用Open3D计算点云的平均曲率作为其几何重要性分数。
    [已修正稳健版]

    Args:
        point_cloud_np (np.ndarray): 一个形状为 [N, 3] 的NumPy数组。

    Returns:
        float: 代表该点云平均曲率的单一分数，错误时返回0.0。
    """
    # 检查1：输入是否有效
    if not isinstance(point_cloud_np, np.ndarray) or point_cloud_np.ndim != 2 or point_cloud_np.shape[1] != 3:
        return 0.0
    # 检查2：点数是否过少
    if point_cloud_np.shape[0] < 30: # 提高阈值，确保邻域搜索有意义
        return 0.0

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

        # 估算法线
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 检查3：是否成功计算出协方差矩阵
        if not pcd.has_covariances():
            return 0.0

        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(pcd.covariances)
        
        # 检查4：特征值计算结果是否有效
        if eigenvalues.shape[0] == 0:
            return 0.0
        
        denominator = np.sum(eigenvalues, axis=1)
        
        # 检查5：分母是否为0，避免除以0的错误
        # 我们将为0的分母替换为一个很小的数，或者直接将这些点的曲率视为0
        curvatures = np.zeros_like(denominator)
        valid_mask = denominator != 0
        curvatures[valid_mask] = eigenvalues[valid_mask, 0] / denominator[valid_mask]
        
        # 计算平均曲率，忽略所有NaN或无穷大的无效值
        mean_curvature = np.nanmean(curvatures)

        # 检查6：最终结果是否是有效的数字
        if not np.isfinite(mean_curvature):
            return 0.0
        
        return float(mean_curvature)

    except Exception as e:
        # 捕获任何其他意外错误
        # print(f"计算曲率时发生意外错误: {e}") # 可以取消注释进行调试
        return 0.0

# --- 主程序：演示和验证该函数的功能 ---
# (验证部分的代码保持不变，此处省略)
if __name__ == '__main__':
    print("--- 基于曲率的重要性计算函数功能验证 ---")
    # ... (之前的验证代码) ...