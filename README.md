# pytorch-week3

## Structure
- `code/resnet_cifar.py` — ResNet-18 CIFAR training & visualization.
- `code/transformer_toy_mt.py` — Minimal Transformer seq2seq toy MT pipeline.
- `runs/cls/` — ResNet outputs (created after running).
- `runs/mt/` — Transformer outputs (created after running).
- `report/one_page_report.md` — Visual summary.

## Requirements
- Python 3.8+
- CUDA GPU recommended
- pip install -r requirements.txt

requirements.txt:
torch torchvision matplotlib scikit-learn numpy

## How to run (example)
1. ResNet:
   - `python code/resnet_cifar.py --epochs 100 --batch-size 128 --lr 0.1`
   - artifacts → `runs/cls/`

2. Transformer:
   - `python code/transformer_toy_mt.py --epochs 40 --batch-size 64 --d-model 128`
   - artifacts → `runs/mt/`

Adjust epochs and batch-size for available GPU memory.
