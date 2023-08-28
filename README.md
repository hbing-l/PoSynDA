# PoSynDA: Multi-Hypothesis Pose Synthesis Domain Adaptation for Robust 3D Human Pose Estimation


PoSynDA is a novel framework for 3D Human Pose Estimation (3D HPE) that addresses the challenges of adapting to new datasets due to the scarcity of 2D-3D pose pairs in target domain training sets. This repository contains the official PyTorch implementation of the PoSynDA method as described in our paper.

## Key Features

- **Domain-Adaptation**: Generative, target-specific source augmentation with a multi-hypothesis approach.
- **Optimization Strategy**: Teacher-student learning paradigm for efficient model training.
- **Efficient Domain Adaptation**: Low-rank adaptation for fine-tuning.

## Prerequisites

- Python 3.x
- PyTorch >= 0.4.0
- einops
- timm
- tensorboard
- [Other dependencies](requirements.txt)

You should download [MATLAB](https://www.mathworks.com/products/matlab-online.html) if you want to evaluate our model on MPI-INF-3DHP dataset.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/hbing-l/PoSynDA.git
   cd PoSynDA
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset Preparation

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

- **Human3.6M**: [Instructions and processed data link](./data/Human3.6M/README.md)
- **MPI-INF-3DHP**: [Instructions and processed data link](./data/3DHP/README.md)


## Training

h36m_transfer.py is the code to transfer H36M S1 to S5, S6, S7, S8, and h36m_3dhp_transfer.py is the code to transfer H36M dataset to 3DHP dataset. To train the PoSynDA model on the target dataset (e.g. 3DHP), run:

```
python h36m_3dhp_transfer.py -k cpn_ft_h36m_dbb -num_proposals 3 -timestep 1000 -c checkpoint/ -gpu 0 --nolog

```


## Evaluation

For evaluation of the provided model. 

```
python h36m_3dhp_transfer.py -c checkpoint -gpu 0 --nolog --evaluate best_epoch.bin
```

## Results

Our method achieves a 58.2mm MPJPE on the Human3.6M dataset without using 3D labels from the target domain, comparable to the target-specific MixSTE model (58.2mm vs. 57.9mm).



## Citation

If you find this work useful for your research, please consider citing our paper:

```
@article{liu2023posynda,
  title={PoSynDA: Multi-Hypothesis Pose Synthesis Domain Adaptation for Robust 3D Human Pose Estimation},
  author={Liu, Hanbing and He, Jun-Yan and Cheng, Zhi-Qi and Xiang, Wangmeng and Yang, Qize and Chai, Wenhao and Wang, Gaoang and Bao, Xu and Luo, Bin and Geng, Yifeng and others},
  journal={arXiv preprint arXiv:2308.09678},
  year={2023}
}
```

## Acknowledgments

We would like to thank all the following contributors and researchers who made this project possible.
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)
* [video-to-pose3D](https://github.com/zh-plus/video-to-pose3D)
* [D3DP](https://github.com/paTRICK-swk/D3DP/tree/main)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.














