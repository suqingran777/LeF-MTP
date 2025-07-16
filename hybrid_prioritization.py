# hybrid_prioritization.py (最终版 - 已修正所有已知错误)
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# *** 核心修正部分：从sklearn中导入所有需要的工具 ***
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# 导入我们自己编写的模块
from dataloaders import get_scanobjectnn_dataloaders
from model import DGCNN

def calculate_fault_score(logits: torch.Tensor, method: str = 'least_confidence') -> torch.Tensor:
    """计算基于不确定性的故障分数。"""
    probabilities = F.softmax(logits, dim=-1)
    if method == 'least_confidence':
        most_confident_prob, _ = torch.max(probabilities, dim=-1)
        return 1.0 - most_confident_prob
    else:
        raise ValueError(f"未知的方法: {method}")

def print_fault_details(df: pd.DataFrame, fault_indices: set, title: str):
    """打印指定索引的故障样本的详细信息"""
    print(f"\n{title}")
    if not fault_indices:
        print("    （未发现此类故障样本）")
        return
    details_df = df.loc[list(fault_indices)].sort_values(by='hybrid_score', ascending=False)
    print(details_df.to_string())


if __name__ == '__main__':
    # --- 1. 设置参数 ---
    BATCH_SIZE = 64
    KNN_K = 10
    MODEL_PATH = 'dgcnn_scanobjectnn.pt' 
    FAULT_SCORE_METHOD = 'least_confidence'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}\n")

    # --- 2. 准备数据和加载模型 ---
    print("--- 步骤1: 加载数据和预训练模型 ---")
    _, test_loader = get_scanobjectnn_dataloaders(batch_size=BATCH_SIZE, num_workers=4)
    model = DGCNN(num_classes=test_loader.dataset.num_classes).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"模型 '{MODEL_PATH}' 加载成功。\n")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。请先使用 train.py 训练并保存一个模型。")
        sys.exit(1)
    model.eval()

    # --- 3. 提取特征并计算基础分数 ---
    print("--- 步骤2: 提取特征并计算基础分数 ---")
    all_features, all_labels, all_logits, sample_ids = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="提取特征")):
            data = data.to(device)
            logits, features = model(data, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(data.y.cpu())
            all_logits.append(logits.cpu())
            for j in range(data.num_graphs):
                sample_ids.append(f'batch_{i}_item_{j}')
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    fault_scores = calculate_fault_score(all_logits).numpy()
    
    print("正在计算K近邻...")
    knn = NearestNeighbors(n_neighbors=KNN_K + 1, metric='cosine').fit(all_features)
    _, indices = knn.kneighbors(all_features)
    
    importance_scores = []
    for i in tqdm(range(len(all_labels)), desc="计算易混淆性"):
        current_label = all_labels[i]
        neighbor_indices = indices[i][1:]
        neighbor_labels = all_labels[neighbor_indices]
        dissimilar_neighbors = (neighbor_labels != current_label).sum().item()
        importance_scores.append(dissimilar_neighbors / KNN_K)
    importance_scores = np.array(importance_scores)
    print("基础分数计算完成。\n")

    # --- 4. 训练融合模型 ---
    print("--- 步骤3: 训练融合模型 ---")
    pred_labels = all_logits.argmax(dim=-1)
    df = pd.DataFrame({
        'fault_score': fault_scores,
        'importance_score': importance_scores,
        'is_fault': (all_labels != pred_labels).numpy()
    }, index=sample_ids)
    
    X = df[['fault_score', 'importance_score']].values
    y = df['is_fault'].values
    
    X_train, X_test, y_train, y_test, df_train_idx, df_test_idx = train_test_split(
        X, y, df.index, test_size=0.3, random_state=42, stratify=y)
    
    print(f"将使用 {len(X_train)} 个样本训练融合模型...")
    fusion_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    fusion_model.fit(X_train, y_train)
    print("融合模型训练完成。\n")

    # --- 5. 使用融合模型预测并进行多样性重排 ---
    print("--- 步骤4: 预测优先级并进行多样性重排 ---")
    df['hybrid_score'] = fusion_model.predict_proba(X)[:, 1]
    
    top_k = int(len(df) * 0.20)
    top_candidates_df = df.sort_values(by='hybrid_score', ascending=False).head(top_k)
    
    candidate_indices = [df.index.get_loc(i) for i in top_candidates_df.index]
    candidate_features = all_features[candidate_indices]
    
    kmeans = KMeans(n_clusters=20, random_state=0, n_init='auto').fit(candidate_features)
    top_candidates_df = top_candidates_df.copy()
    top_candidates_df['cluster'] = kmeans.labels_
    
    reranked_indices, grouped = [], top_candidates_df.groupby('cluster')
    max_len = grouped.size().max()
    for i in range(max_len):
        for _, group in grouped:
            if i < len(group):
                reranked_indices.append(group.index[i])
    
    remaining_indices = df.sort_values(by='hybrid_score', ascending=False).iloc[top_k:].index
    final_ranked_indices = reranked_indices + remaining_indices.tolist()
    final_reranked_df = df.loc[final_ranked_indices]
    print("多样性重排完成。\n")

    # --- 6. 最终结果报告 ---
    print("--- 步骤5: 最终性能报告与分析 ---\n")
    total_faults = df['is_fault'].sum()
    print(f"测试集中总共存在 {total_faults} 个故障样本。")
    
    ranked_by_fault = df.sort_values(by='fault_score', ascending=False)
    ranked_by_importance = df.sort_values(by='importance_score', ascending=False)

    for p in [0.05, 0.10, 0.20, 0.30]:
        top_n_count = int(len(df) * p)
        faults_s1 = ranked_by_fault.head(top_n_count)['is_fault'].sum()
        faults_s2 = ranked_by_importance.head(top_n_count)['is_fault'].sum()
        faults_s3 = final_reranked_df.head(top_n_count)['is_fault'].sum()

        print(f"\n--- 排序效率对比 (检查前 {p*100:.0f}%) ---")
        print(f"策略1 (仅故障分数/不确定性):     找到了 {faults_s1} / {total_faults} 个故障。")
        print(f"策略2 (仅重要性分数/易混淆性):   找到了 {faults_s2} / {total_faults} 个故障。")
        print(f"策略3 (学习融合+多样性重排):     找到了 {faults_s3} / {total_faults} 个故障。")
        
    top_n_count = int(len(df) * 0.20)
    faults_by_S1 = set(ranked_by_fault[ranked_by_fault['is_fault']].head(top_n_count).index)
    faults_by_S2 = set(ranked_by_importance[ranked_by_importance['is_fault']].head(top_n_count).index)
    common_faults = faults_by_S1.intersection(faults_by_S2)
    unique_to_S1 = faults_by_S1 - faults_by_S2
    unique_to_S2 = faults_by_S2 - faults_by_S1
    
    print(f"\n在排名前20%的样本中：")
    print(f"策略1 (仅故障分数) 独立发现了 {len(unique_to_S1)} 个故障。")
    print(f"策略2 (仅重要性分数) 独立发现了 {len(unique_to_S2)} 个故障。")
    print(f"两种策略共同发现了 {len(common_faults)} 个故障。")
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    print_fault_details(df, unique_to_S1, "--- 仅“故障分数”策略独立发现的故障详情 (全部样本) ---")
    print_fault_details(df, unique_to_S2, "--- 仅“重要性分数”策略独立发现的故障详情 (全部样本) ---")
    print_fault_details(df, common_faults, "--- 两种策略共同发现的故障详情 (全部样本) ---")