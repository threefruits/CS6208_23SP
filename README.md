# CS6208 Project 1: Paper Review and Implementation of Neural Execution of Graph Algorithms
Original Paper: Neural Execution of Graph Algorithms (https://arxiv.org/abs/1910.10593)

## Setup
Clone the main repo
```
https://github.com/threefruits/CS6208_23spring.git
```

Create conda environment and install the dependencies
```
$ pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

$ pip install torch_geometric

$ pip install networkx==2.3 matplotlib==2.2.3 numpy==1.20.0

# Optional dependencies:
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

## Train and Test

Run the Jupyter notebook [train_and_test.ipynb](train_and_test.ipynb). You can also directly run `train.py` and `test.py`.


# References
Veličković, Petar, Rex Ying, Matilde Padovano, Raia Hadsell, and Charles Blundell. "Neural execution of graph algorithms." arXiv preprint arXiv:1910.10593 (2019).