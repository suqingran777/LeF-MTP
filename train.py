# train.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

# 从我们之前的文件中导入函数和类
from dataloaders import get_scanobjectnn_dataloaders
from model import DGCNN

def train_one_epoch(model, train_loader, optimizer, device):
    """
    对模型进行一轮完整的训练。
    """
    model.train()  # 将模型设置为训练模式
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for data in tqdm(train_loader, desc="训练中"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        total_correct += pred.eq(data.y).sum().item()
        total_samples += data.num_graphs
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def evaluate(model, test_loader, device):
    """
    在测试集上评估模型的性能。
    """
    model.eval()  # 将模型设置为评估模式
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            loss = F.cross_entropy(out, data.y)
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            total_correct += pred.eq(data.y).sum().item()
            total_samples += data.num_graphs
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# --- 主程序 ---
if __name__ == '__main__':
    # --- 1. 设置超参数 ---
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    # 您可以根据需要增加训练轮次以获得更好的模型性能
    EPOCHS = 50 
    MODEL_SAVE_PATH = 'dgcnn_scanobjectnn.pt' # 定义模型保存的路径和文件名

    # --- 2. 准备设备、数据和模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}\n")

    # 我们使用ScanObjectNN数据集进行训练
    train_loader, test_loader = get_scanobjectnn_dataloaders(
        batch_size=BATCH_SIZE, 
        num_workers=4
    )

    model = DGCNN(num_classes=train_loader.dataset.num_classes).to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 用于记录最佳模型准确率的变量
    best_test_acc = 0.0

    # --- 3. 开始训练循环 ---
    print("--- 开始正式训练 ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:02d} | 训练损失: {train_loss:.4f}, 训练准确率: {train_acc*100:.2f}% | "
              f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc*100:.2f}%")
        
        # 检查是否是目前最好的模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # 如果是，则保存当前的模型权重
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✨ 新的最佳模型已保存！测试准确率: {best_test_acc*100:.2f}%")
        
    print("\n--- 训练完成 ---")
    print(f"整个训练过程中，最佳测试准确率为: {best_test_acc*100:.2f}%")
    print(f"最佳模型已保存至: {MODEL_SAVE_PATH}")