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

- **Human3.6M**: [Instructions and processed data link](./data/Human3.6M/README.md)
- **MPI-INF-3DHP**: [Instructions and processed data link](./data/MPI-INF-3DHP/README.md)

## Training

To train the PoSynDA model on your dataset, run:

```
python train.py --config path_to_config_file
```

## Evaluation

For evaluation on the provided datasets:

```
python evaluate.py --config path_to_config_file
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

We would like to thank all the contributors and researchers who made this project possible.
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)
* [video-to-pose3D](https://github.com/zh-plus/video-to-pose3D)
* [D3DP](https://github.com/paTRICK-swk/D3DP/tree/main)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.








### Human3.6M

We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).  You can download the processed data from [here](https://drive.google.com/file/d/1FMgAf_I04GlweHMfgUKzB0CMwglxuwPe/view?usp=sharing).  `data_2d_h36m_gt.npz` is the ground truth of 2D keypoints. `data_2d_h36m_cpn_ft_h36m_dbb.npz` is the 2D keypoints obatined by [CPN](https://github.com/GengDavid/pytorch-cpn).  `data_3d_h36m.npz` is the ground truth of 3D human joints. Put them in the `./data` directory.


### MPI-INF-3DHP

We set up the MPI-INF-3DHP dataset following [P-STMO](https://github.com/paTRICK-swk/P-STMO). However, our training/testing data is different from theirs. They train and evaluate on 3D poses scaled to the height of the universal skeleton used by Human3.6M (officially called "univ_annot3"), while we use the ground truth 3D poses (officially called "annot3"). The former does not guarantee that the reprojection (used by the proposed JPMA) of the rescaled 3D poses is consistent with the 2D inputs, while the latter does. You can download our processed data from [here](https://drive.google.com/file/d/1zOM_CvLr4Ngv6Cupz1H-tt1A6bQPd_yg/view?usp=share_link). Put them in the `./data` directory. 

## Evaluate

h36m_transfer.py is the code to transfer H36M S1 to S5, S6, S7, S8, and h36m_3dhp_transfer.py is the code to transfer H36M dataset to 3DHP dataset.




