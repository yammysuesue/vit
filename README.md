# vit

使用 DINOv2 / Vision Transformer 的图像分类项目。

原 `hw4` 文件夹中的内容已直接放在本仓库根目录。

- `data/`：训练与测试图像。
- `dinov2/`：DINOv2 源码及其原有许可证。
- `weight/`：预训练模型权重。
- `output_stage1/`：训练权重、日志、预测结果和可视化。
- `error_analysis/`：分类错误分析。
- `train_dinov2_stage1.py`：训练脚本。
- `infer_dinov2.py`、`infer_dinov2.ipynb`：推理脚本与笔记本。
- `visualize_dinov2_metrics.py`：指标可视化脚本。

模型权重（`*.pth`）使用 Git LFS 存储。安装 Git LFS 后下载：

```bash
git lfs install
git clone https://github.com/yammysuesue/vit.git
cd vit
git lfs pull
```

PPT 文件及 Python / Jupyter 缓存不纳入版本管理。
