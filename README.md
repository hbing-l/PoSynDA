# PoSynDA: Multi-Hypothesis Pose Synthesis Domain Adaptation for Enhanced 3D Human Pose Estimation

The PyTorch implementation for ["PoSynDA: Multi-Hypothesis Pose Synthesis Domain Adaptation for Enhanced 3D Human Pose Estimation"].
<p align="center"><img src="fig/overview.jpg", width="600" alt="" /></p>
<p align="center"><img src="fig/demo.gif", width="600"  alt="" /></p>


## Dependencies

Make sure you have the following dependencies installed (python):

* pytorch >= 0.4.0
* matplotlib=3.1.0
* einops
* timm
* tensorboard

You should download [MATLAB](https://www.mathworks.com/products/matlab-online.html) if you want to evaluate our model on MPI-INF-3DHP dataset.

## Datasets

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

### Human3.6M

We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).  You can download the processed data from [here](https://drive.google.com/file/d/1FMgAf_I04GlweHMfgUKzB0CMwglxuwPe/view?usp=sharing).  `data_2d_h36m_gt.npz` is the ground truth of 2D keypoints. `data_2d_h36m_cpn_ft_h36m_dbb.npz` is the 2D keypoints obatined by [CPN](https://github.com/GengDavid/pytorch-cpn).  `data_3d_h36m.npz` is the ground truth of 3D human joints. Put them in the `./data` directory.

### MPI-INF-3DHP

We set up the MPI-INF-3DHP dataset following [P-STMO](https://github.com/paTRICK-swk/P-STMO). However, our training/testing data is different from theirs. They train and evaluate on 3D poses scaled to the height of the universal skeleton used by Human3.6M (officially called "univ_annot3"), while we use the ground truth 3D poses (officially called "annot3"). The former does not guarantee that the reprojection (used by the proposed JPMA) of the rescaled 3D poses is consistent with the 2D inputs, while the latter does. You can download our processed data from [here](https://drive.google.com/file/d/1zOM_CvLr4Ngv6Cupz1H-tt1A6bQPd_yg/view?usp=share_link). Put them in the `./data` directory. 

## Evaluate

h36m_transfer.py is the code to transfer H36M S1 to S5, S6, S7, S8, and h36m_3dhp_transfer.py is the code to transfer H36M dataset to 3DHP dataset.

## Acknowledgement
Our code refers to the following repositories.
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)
* [video-to-pose3D](https://github.com/zh-plus/video-to-pose3D)
* [D3DP](https://github.com/paTRICK-swk/D3DP/tree/main)

We thank the authors for releasing their codes.