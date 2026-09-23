# vit

An image classification project using DINOv2 and Vision Transformers.

The project files are located directly in the repository root.

- `data/`: Training and test images.
- `dinov2/`: DINOv2 source code and its original licenses.
- `weight/`: Pretrained model weights.
- `output_stage1/`: Trained model weights, logs, predictions, and visualizations.
- `error_analysis/`: Classification error analysis.
- `train_dinov2_stage1.py`: Training script.
- `infer_dinov2.py` and `infer_dinov2.ipynb`: Inference script and notebook.
- `visualize_dinov2_metrics.py`: Metric visualization script.

Model weights (`*.pth`) are stored with Git LFS. After installing Git LFS, download the project with:

```bash
git lfs install
git clone https://github.com/yammysuesue/vit.git
cd vit
git lfs pull
```

PowerPoint files and generated Python / Jupyter caches are excluded from version control.
