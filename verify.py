try:
    import torch
    import torch_geometric
    import pyg_lib
    import torch_scatter
    import torch_sparse
    import pandas
    import open3d

    print("✅ 成功！所有核心库都已正确导入。")
    print(f"   - PyTorch 版本: {torch.__version__}")
    print(f"   - PyG 版本: {torch_geometric.__version__}")

    if torch.cuda.is_available():
        print(f"✅ GPU 已找到: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ 错误: 未找到GPU。")

except Exception as e:
    print(f"❌ 验证失败: {e}")