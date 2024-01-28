# Multi-Granularity Sequence Generation for Hierarchical Image Classification
Official PyTorch implementation of [Multi-Granularity Sequence Generation for Hierarchical Image Classification](https://link.springer.com/article/10.1007/s41095-022-0332-2)

## 1. Run 

python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
--cfg configs/swin_large_patch4_window12_384_22kto1k_finetune.yaml --pretrained swin_large_patch4_window12_384_22k.pth \
--batch-size 16

## 2. Citation

@article{liu2024multi,
  title={Multi-granularity sequence generation for hierarchical image classification},
  author={Liu, Xinda and Wang, Lili},
  journal={Computational Visual Media},
  volume={10},
  number={2},
  pages={243--260},
  year={2024},
  publisher={Springer}
}
