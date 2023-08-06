import itertools
import numpy as np
import random

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

from common.diffusionpose_3dhp_transfer import *

from common.loss import *
from common.generators_3dhp import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from common.generators import ChunkedGenerator_Seq1, UnchunkedGenerator_Seq1
from common.weight import TransformerModel
from common.lora import inject_trainable_lora, extract_lora_ups_down
from time import time
from common.utils import *
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from itertools import cycle
import scipy.io as scio

#cudnn.benchmark = True       
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if args.evaluate != '':
    description = "Evaluate!"
elif args.evaluate == '':
    description = "Train!"

# initial setting
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
# tensorboard
if not args.nolog:
    writer = SummaryWriter(args.log+'_'+TIMESTAMP)
    writer.add_text('description', description)
    writer.add_text('command', 'python ' + ' '.join(sys.argv))
    # logging setting
    logfile = os.path.join(args.log+'_'+TIMESTAMP, 'logging.log')
    sys.stdout = Logger(logfile)
print(description)
print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)

manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# if not assign checkpoint path, Save checkpoint file into log folder
if args.checkpoint=='':
    args.checkpoint = args.log+'_'+TIMESTAMP
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

# dataset loading
print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing target data...')

out_poses_3d_train = {}
out_poses_2d_train = {}
out_poses_3d_test = {}
out_poses_2d_test = {}
valid_frame = {}

kps_left, kps_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
joints_left, joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]


data_train = np.load("./data/data_train_3dhp_ori.npz", allow_pickle=True)['data'].item()
for seq in data_train.keys():
    for cam in data_train[seq][0].keys():
        anim = data_train[seq][0][cam]

        subject_name, seq_name = seq.split(" ")

        data_3d = anim['data_3d']
        data_3d[:, :14] -= data_3d[:, 14:15]
        data_3d[:, 15:] -= data_3d[:, 14:15]
        out_poses_3d_train[(subject_name, seq_name, cam)] = data_3d

        data_2d = anim['data_2d']

        data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
        out_poses_2d_train[(subject_name, seq_name, cam)] = data_2d


data_test = np.load("./data/data_test_3dhp_ori.npz", allow_pickle=True)['data'].item()
for seq in data_test.keys():

    anim = data_test[seq]

    valid_frame[seq] = anim["valid"]

    data_3d = anim['data_3d']
    data_3d[:, :14] -= data_3d[:, 14:15]
    data_3d[:, 15:] -= data_3d[:, 14:15]
    out_poses_3d_test[seq] = data_3d

    data_2d = anim['data_2d']

    if seq == "TS5" or seq == "TS6":
        width = 1920
        height = 1080
    else:
        width = 2048
        height = 2048
    data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
    out_poses_2d_test[seq] = data_2d


print('Preparing source data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
source_kps_left, source_kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
source_joints_left, source_joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

###################
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]



def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)  # poses_train: list len 600 poses_train[0].shape=[1383, 17, 3]


# set receptive_field as number assigned
receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Receptive field', str(receptive_field))
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = args.min_loss


model_pos_train = D3DP(args, joints_left, joints_right, is_train=True, num_proposals=args.num_proposals, sampling_timesteps=args.sampling_timesteps)
model_weight_train = TransformerModel(ntoken=3, ninp=512, nhead=8, nhid=512, nlayers=8)
model_pos_test_temp = D3DP(args,joints_left, joints_right, is_train=False)
model_weight_temp = TransformerModel(ntoken=3, ninp=512, nhead=8, nhid=512, nlayers=8)
model_pos = D3DP(args,joints_left, joints_right,  is_train=False)
model_weight = TransformerModel(ntoken=3, ninp=512, nhead=8, nhid=512, nlayers=8)

#################
causal_shift = 0
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Trainable parameter count', str(model_params/1000000) + ' Million')

# make model parallel
if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()
    model_pos_test_temp = nn.DataParallel(model_pos_test_temp)
    model_pos_test_temp = model_pos_test_temp.cuda()
    model_weight = nn.DataParallel(model_weight)
    model_weight = model_weight.cuda()
    model_weight_train = nn.DataParallel(model_weight_train)
    model_weight_train = model_weight_train.cuda()
    model_weight_temp = nn.DataParallel(model_weight_temp)
    model_weight_temp = model_weight_temp.cuda()

checkpoint = torch.load('checkpoint/model_h36m/h36m_best_epoch.bin', map_location=lambda storage, loc: storage)
print('This model was trained for {} epochs'.format(checkpoint['epoch']))
model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)

# freeze the denoiser parameters, inject the lora parameters
model_pos_train.requires_grad_(False)
lora_params, train_names = inject_trainable_lora(model_pos_train)

model_pos_test_temp.requires_grad_(False)
lora_params2, train_names2 = inject_trainable_lora(model_pos_test_temp)

model_pos.requires_grad_(False)
lora_params3, train_names3 = inject_trainable_lora(model_pos)

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    # chk_filename = args.resume or args.evaluate
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

# get target training data
train_test_generator = UnchunkedGenerator_Seq(None, out_poses_3d_test, out_poses_2d_test,
                                    pad=pad, causal_shift=causal_shift, augment=args.data_augmentation,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right, valid_frame=valid_frame)

test_generator = UnchunkedGenerator_Seq(None, out_poses_3d_test, out_poses_2d_test,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right, valid_frame=valid_frame)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Testing Frames', str(test_generator.num_frames()))


# get source training data
source_train_generator = UnchunkedGenerator_Seq1(cameras_train, poses_train, poses_train_2d,
                                            pad=pad, causal_shift=causal_shift, augment=False)


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d, valid_frame):

    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)
    valid_frame = valid_frame.unsqueeze(1)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
        out_num = inputs_2d_p.shape[0] // receptive_field+1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])
    eval_valid_frame = torch.empty(out_num, receptive_field, 1)

    for i in range(out_num-1):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_valid_frame[i, :, :] = valid_frame[i * receptive_field:i * receptive_field + receptive_field, :]
    if inputs_2d_p.shape[0] < receptive_field:
        from torch.nn import functional as F
        pad_right = receptive_field-inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
        # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
        inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
    if valid_frame.shape[0] < receptive_field:
        pad_right = receptive_field-valid_frame.shape[0]
        valid_frame = rearrange(valid_frame, 'f c -> c f')
        valid_frame = F.pad(valid_frame, (0,pad_right), mode='replicate')
        valid_frame = rearrange(valid_frame, 'c f -> f c')
    eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
    eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]
    eval_valid_frame[-1, :, :] = valid_frame[-receptive_field:, :]

    return eval_input_2d, eval_input_3d, eval_valid_frame

def source_eval_data_prepare(receptive_field, inputs_2d, inputs_3d):

    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
        out_num = inputs_2d_p.shape[0] // receptive_field+1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num-1):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
    if inputs_2d_p.shape[0] < receptive_field:
        from torch.nn import functional as F
        pad_right = receptive_field-inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
        # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
        inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
    eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
    eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]

    return eval_input_2d, eval_input_3d


def pose_post_process(pose_pred, data_list, keys, receptive_field):
    for ii in range(pose_pred.shape[0] - 1):
        data_list[keys][:, ii * receptive_field:(ii + 1) * receptive_field] = pose_pred[ii]
    data_list[keys][:, -receptive_field:] = pose_pred[-1]
    data_list[keys] = data_list[keys].transpose(3, 2, 1, 0)
    return data_list

def cam_mm_to_pix(cam, cam_data):
    # w, h, ss_x, ss_y
    mx = cam_data[0] / cam_data[2]
    my = cam_data[1] / cam_data[3]
    cam[0] = cam[0] * mx
    cam[1] = cam[1] * my
    cam[2] = cam[2] * mx + cam_data[0]/2
    cam[3] = cam[3] * my + cam_data[1]/2

    return cam
###################

def global_position_redistribution(source_3d, target_2d, camera_params):
    """
    global position re-distribution torch version
    """

    source_3d = source_3d.detach()
    
    fx, fy = camera_params[0, :2]
    if source_3d.shape[0] <= target_2d.shape[0]:
        index = torch.randperm(source_3d.shape[0])
        target_2d = target_2d[index]
        camera_params = camera_params.repeat(source_3d.shape[0], 1)
    else:
        index = torch.randperm(target_2d.shape[0])
        source_3d = source_3d[index]
        camera_params = camera_params.repeat(target_2d.shape[0], 1)

    # calculate 2d scale
    w = torch.max(target_2d[..., 0], dim=-1)[0] - torch.min(target_2d[..., 0], dim=-1)[0]
    h = torch.max(target_2d[..., 1], dim=-1)[0] - torch.min(target_2d[..., 1], dim=-1)[0]
    s = (w + h) / 2

    # calculate 3d range
    dx = torch.max(source_3d[..., 0], dim=-1)[0] - torch.min(source_3d[..., 0], dim=-1)[0]
    dy = torch.max(source_3d[..., 1], dim=-1)[0] - torch.min(source_3d[..., 1], dim=-1)[0]

    # calculate z
    z = (fx * dx + fy * dy) / (2 * s)

    # process with camera params
    target_2d[..., 0, :] -= camera_params[..., 2:4] # c
    target_2d[..., 0, :] /= camera_params[..., :2] # f

    u, v = target_2d[..., 0, 0], target_2d[..., 0, 1]

    # calculate x
    x, y = z * u, z * v

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    position = torch.stack([x,y,z], axis=1).reshape(-1, 1, 3)

    source_3d = source_3d - source_3d[:, :1, :] + position

    return source_3d, camera_params


# Training start
if not args.evaluate:

    lr = args.learning_rate
    optimizer = optim.Adam(itertools.chain(*lora_params, model_weight_train.parameters()), lr=1e-4)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_pos_train = []
    losses_2d_pos_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []
    losses_3d_depth_valid = []

    epoch = 0
    best_epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    print('INFO: Training on {} frames'.format(train_test_generator.num_frames()))
    if not args.nolog:
        writer.add_text(args.log+'_'+TIMESTAMP + '/Training Frames', str(train_test_generator.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        if not args.coverlr:
            lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')

    # Pos model only
    while epoch < args.epochs:
        
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_3d_pos_train = 0
        epoch_loss_2d_pos_train = 0
        epoch_loss_3d_pos_true = 0
        model_pos_train.train()
        model_weight_train.train()
        iteration = 0

        num_frames = train_test_generator.num_frames()
        
        N = 0
        data_inference_all = {}
        data_inference_mean = {}
        data_inference_h_min = {}
        data_inference_joint_min = {}
        data_inference_reproj_min = {}

        cam_1 = torch.tensor([7.32506, 7.32506, -0.0322884, 0.0929296, 0, 0, 0, 0, 0])
        cam_data_1 = [2048, 2048, 10, 10] #width, height, sensorSize_x, sensorSize_y
        cam_2 = torch.tensor([8.770747185, 8.770747185, -0.104908645, 0.104899704, 0, 0, 0, 0, 0])
        cam_data_2 = [1920, 1080, 10, 5.625]  # width, height, sensorSize_x, sensorSize_y
        cam_1 = cam_mm_to_pix(cam_1, cam_data_1)
        cam_2 = cam_mm_to_pix(cam_2, cam_data_2)

        quickdebug=args.debug
        
        for ((_, batch, batch_2d, batch_valid, keys ), (_, src_batch_3d, src_batch_2d))in zip(train_test_generator.next_epoch(), cycle(source_train_generator.next_epoch())):

            if iteration % 1000 == 0:
                print("%d/%d"% (iteration, num_frames))
            
            # target   
            
            tar_inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            tar_inputs_3d = torch.from_numpy(batch.astype('float32'))
            tar_inputs_valid = torch.from_numpy(batch_valid.astype('float32'))

            _, f_sz, j_sz, c_sz = tar_inputs_3d.shape
            
            print(keys)

            ##### apply test-time-augmentation (following Videopose3d)
            tar_inputs_2d_flip = tar_inputs_2d.clone()
            tar_inputs_2d_flip [:, :, :, 0] *= -1
            tar_inputs_2d_flip[:, :, kps_left + kps_right,:] = tar_inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            tar_inputs_3d_p = tar_inputs_3d
            
            tar_inputs_2d, tar_inputs_3d, tar_valid_frame = eval_data_prepare(receptive_field, tar_inputs_2d, tar_inputs_3d_p, inputs_valid)
            tar_inputs_2d_flip, _, _ = eval_data_prepare(receptive_field, tar_inputs_2d_flip, tar_inputs_3d_p, tar_inputs_valid)

            if torch.cuda.is_available():
                tar_inputs_2d = tar_inputs_2d.cuda()
                tar_inputs_2d_flip = tar_inputs_2d_flip.cuda()
                tar_inputs_3d = tar_inputs_3d.cuda()  # [26, 243, 17, 3]

            
            bs = 2
            total_batch = (tar_inputs_3d.shape[0] + bs - 1) // bs

            for batch_cnt in range(total_batch):

                if (batch_cnt + 1) * bs > tar_inputs_3d.shape[0]:
                    tar_inputs_2d_single = tar_inputs_2d[batch_cnt * bs:]
                    tar_inputs_2d_flip_single = tar_inputs_2d_flip[batch_cnt * bs:]
                    tar_inputs_3d_single = tar_inputs_3d[batch_cnt * bs:]
                    # valid_frame_single = valid_frame[batch_cnt * bs:]
                else:
                    tar_inputs_2d_single = tar_inputs_2d[batch_cnt * bs:(batch_cnt+1) * bs]
                    tar_inputs_2d_flip_single = tar_inputs_2d_flip[batch_cnt * bs:(batch_cnt+1) * bs]
                    tar_inputs_3d_single = tar_inputs_3d[batch_cnt * bs:(batch_cnt+1) * bs]
                    # valid_frame_single = valid_frame[batch_cnt * bs:(batch_cnt + 1) * bs]

                traj = tar_inputs_3d_single[:, :, 14:15].clone()
                tar_inputs_3d_single[:, :, 14] = 0  # [2, 243, 17, 3]

                optimizer.zero_grad()
                
                predicted_3d_pos_single = model_pos_train(tar_inputs_2d_single, tar_inputs_3d_single, input_2d_flip=tar_inputs_2d_flip_single) #b, t, h, f, j, c
                b,t,h,f,j,c = predicted_3d_pos_single.shape
                predicted_3d_pos_single = predicted_3d_pos_single.reshape(-1, j, c)
                predicted_3d_pos_single = model_weight_train(predicted_3d_pos_single)
                predicted_3d_pos_single = predicted_3d_pos_single.reshape(b, t, h, f, j, c)
                
                predicted_3d_pos_single[:, :, :, :, 14] = 0

                # find the best 3d pose for target input
                inputs_traj_single_all = traj.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
                predicted_3d_pos_abs_single = predicted_3d_pos_single + inputs_traj_single_all
                #predicted_3d_pos_abs_single = predicted_3d_pos_abs_single/1000
                predicted_3d_pos_abs_single = predicted_3d_pos_abs_single.reshape(b * t * h * f, j_sz, c_sz)
                if keys == "TS5" or keys == "TS6":
                    cam = cam_2.clone()
                    cam_data = cam_data_2.copy()
                    reproject_func = project_to_2d
                else:
                    cam = cam_1.clone()
                    cam_data = cam_data_1.copy()
                    reproject_func = project_to_2d_linear

                cam_single_all = cam.unsqueeze(0).repeat(b * t * h * f, 1).cuda()
                reproj_2d = reproject_func(predicted_3d_pos_abs_single, cam_single_all)
                reproj_2d = reproj_2d.reshape(b, t, h, f, j_sz, 2)

                cam_single_gt = cam.unsqueeze(0).repeat(b * f, 1).cuda()
                input_3d_reproj = inputs_3d_single + traj
                input_3d_reproj = input_3d_reproj.reshape(b * f, 17, 3)
                reproj_2d_gt = reproject_func(input_3d_reproj, cam_single_gt)
                reproj_2d_gt = reproj_2d_gt.reshape(b, f, 17, 2)

                target_2d = torch.from_numpy(image_coordinates(tar_inputs_2d_single[..., :2].cpu().numpy(), w=cam_data[0], h=cam_data[1])).cuda()
                target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
                errors_2d = torch.norm(reproj_2d - target_2d, dim=len(target_2d.shape) - 1) # b, t, h, f, n
                reproj_min_indices = torch.min(errors_2d, dim=2, keepdim=True).indices  # b,t,1,f,n
                reproj_min_indices = reproj_min_indices.unsqueeze(-1).repeat(1, 1, 1, 1, 1, c_sz)
                reproj_min_pose = torch.gather(predicted_3d_pos_single, 2, reproj_min_indices).squeeze(2)

                reproj_all_min_pose = reproj_min_pose.unsqueeze(2).repeat(1, 1, h, 1, 1, 1)
                loss_2d = torch.mean(torch.min(errors_2d, dim=2, keepdim=True).values)
                
                loss_3d = mpjpe(reproj_all_min_pose, predicted_3d_pos_single)
                
                lambd1 = 0.5
                lambd2 = 1
                loss_total = lambd1 * loss_2d + lambd2 * loss_3d
                
                loss_total.backward(loss_total.clone().detach())

                loss_total = torch.mean(loss_total)

                epoch_loss_3d_train += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * loss_total.item()
                epoch_loss_3d_pos_train += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * loss_3d.item()
                epoch_loss_2d_pos_train += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * loss_2d.item()
                N += inputs_3d_single.shape[0] * inputs_3d_single.shape[1]

                optimizer.step()
                
                            
            # source
            src_inputs_3d = torch.from_numpy(src_batch_3d.astype('float32'))
            src_inputs_2d = torch.from_numpy(src_batch_2d.astype('float32'))
            # print("source shape ", src_inputs_3d.shape)
            
            b, f, n, c = src_inputs_3d.shape
            source_3d = src_inputs_3d.reshape(b*f, n, c)
            tb, tf, tn, tc = inputs_2d.shape
            target_2d = inputs_2d.reshape(tb*tf, tn, tc).cpu()
            if keys == "TS5" or keys == "TS6":
                cam = cam_2.clone()
                cam_data = cam_data_2.copy()
                reproject_func = project_to_2d
            else:
                cam = cam_1.clone()
                cam_data = cam_data_1.copy()
                reproject_func = project_to_2d_linear
                
            camera_params = cam.unsqueeze(0)
            source_3d, camera_params = global_position_redistribution(source_3d, target_2d, camera_params)
            reproj_2d = reproject_func(source_3d, camera_params)
            src_inputs_2d = reproj_2d.reshape(b, -1, n, 2).float()
            src_inputs_3d = source_3d.reshape(b, -1, n, 3).float()
            # print("transform source shape ", src_inputs_3d.shape)

            ##### apply test-time-augmentation (following Videopose3d)
            src_inputs_2d_flip = src_inputs_2d.clone()
            src_inputs_2d_flip[:, :, :, 0] *= -1
            src_inputs_2d_flip[:, :, source_kps_left + source_kps_right, :] = src_inputs_2d_flip[:, :, source_kps_right + source_kps_left, :]

            ##### convert size
            src_inputs_3d_p = src_inputs_3d
            src_inputs_2d, src_inputs_3d = source_eval_data_prepare(receptive_field, src_inputs_2d, src_inputs_3d_p)
            src_inputs_2d_flip, _ = source_eval_data_prepare(receptive_field, src_inputs_2d_flip, src_inputs_3d_p)

            if torch.cuda.is_available():
                src_inputs_3d = src_inputs_3d.cuda()   # [6, 243, 17, 3]
                src_inputs_2d = src_inputs_2d.cuda()   # [6, 243, 17, 2]
                src_inputs_2d_flip = src_inputs_2d_flip.cuda()
            src_inputs_3d[:, :, 0] = 0
            
            inputs_3d = src_inputs_3d
            inputs_2d = src_inputs_2d
            inputs_2d_flip = src_inputs_2d_flip

            bs = 2
            total_batch = (inputs_3d.shape[0] + bs - 1) // bs

            for batch_cnt in range(total_batch):

                if (batch_cnt + 1) * bs > inputs_3d.shape[0]:
                    inputs_2d_single = inputs_2d[batch_cnt * bs:]
                    inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:]
                    inputs_3d_single = inputs_3d[batch_cnt * bs:]
                    # valid_frame_single = valid_frame[batch_cnt * bs:]
                else:
                    inputs_2d_single = inputs_2d[batch_cnt * bs:(batch_cnt+1) * bs]
                    inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:(batch_cnt+1) * bs]
                    inputs_3d_single = inputs_3d[batch_cnt * bs:(batch_cnt+1) * bs]
                    # valid_frame_single = valid_frame[batch_cnt * bs:(batch_cnt + 1) * bs]

                traj = inputs_3d_single[:, :, 14:15].clone()
                inputs_3d_single[:, :, 14] = 0  # [2, 243, 17, 3]

                optimizer.zero_grad()
                
                predicted_3d_pos_single = model_pos_train(inputs_2d_single, inputs_3d_single, input_2d_flip=inputs_2d_flip_single) #b, t, h, f, j, c
                b,t,h,f,j,c = predicted_3d_pos_single.shape
                predicted_3d_pos_single = predicted_3d_pos_single.reshape(-1, j, c)
                predicted_3d_pos_single = model_weight_train(predicted_3d_pos_single)
                predicted_3d_pos_single = predicted_3d_pos_single.reshape(b, t, h, f, j, c)
                
                predicted_3d_pos_single[:, :, :, :, 14] = 0

                inputs_traj_single_all = traj.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
                predicted_3d_pos_abs_single = predicted_3d_pos_single + inputs_traj_single_all
                predicted_3d_pos_abs_single = predicted_3d_pos_abs_single.reshape(b * t * h * f, j_sz, c_sz)
                if keys == "TS5" or keys == "TS6":
                    cam = cam_2.clone()
                    cam_data = cam_data_2.copy()
                    reproject_func = project_to_2d
                else:
                    cam = cam_1.clone()
                    cam_data = cam_data_1.copy()
                    reproject_func = project_to_2d_linear

                cam_single_all = cam.unsqueeze(0).repeat(b * t * h * f, 1).cuda()
                reproj_2d = reproject_func(predicted_3d_pos_abs_single, cam_single_all)
                reproj_2d = reproj_2d.reshape(b, t, h, f, j_sz, 2)
                #reproj_2d[..., :2] = torch.from_numpy(normalize_screen_coordinates(reproj_2d[..., :2].cpu().numpy(), w=cam_data[0],h=cam_data[1])).cuda()

                cam_single_gt = cam.unsqueeze(0).repeat(b * f, 1).cuda()
                input_3d_reproj = inputs_3d_single + traj
                input_3d_reproj = input_3d_reproj.reshape(b * f, 17, 3)
                reproj_2d_gt = reproject_func(input_3d_reproj, cam_single_gt)
                reproj_2d_gt = reproj_2d_gt.reshape(b, f, 17, 2)

                src_2d = torch.from_numpy(image_coordinates(inputs_2d_single[..., :2].cpu().numpy(), w=cam_data[0], h=cam_data[1])).cuda()
                src_2d = src_2d.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
                errors_2d = torch.norm(reproj_2d - src_2d, dim=len(src_2d.shape) - 1) # b, t, h, f, n
                reproj_min_indices = torch.min(errors_2d, dim=2, keepdim=True).indices  # b,t,1,f,n
                reproj_min_indices = reproj_min_indices.unsqueeze(-1).repeat(1, 1, 1, 1, 1, c_sz)
                reproj_min_pose = torch.gather(predicted_3d_pos_single, 2, reproj_min_indices).squeeze(2)

                reproj_all_min_pose = reproj_min_pose.unsqueeze(2).repeat(1, 1, h, 1, 1, 1)
                loss_2d = torch.mean(torch.min(errors_2d, dim=2, keepdim=True).values)

                label = inputs_3d_single.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
                loss_3d = mpjpe(label, predicted_3d_pos_single)
                
                lambd1 = 0.1
                lambd2 = 1
                loss_total = lambd1 * loss_2d + lambd2 * loss_3d
                
                loss_total.backward(loss_total.clone().detach())

                loss_total = torch.mean(loss_total)

                epoch_loss_3d_train += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * loss_total.item()
                epoch_loss_3d_pos_train += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * loss_3d.item()
                epoch_loss_2d_pos_train += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * loss_2d.item()
                N += inputs_3d_single.shape[0] * inputs_3d_single.shape[1]

                optimizer.step()

            iteration += 1
            
        
        losses_3d_train.append(epoch_loss_3d_train / N)
        losses_3d_pos_train.append(epoch_loss_3d_pos_train / N)
        losses_2d_pos_train.append(epoch_loss_2d_pos_train / N)
                
        with torch.no_grad():
            model_pos_test_temp.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos_test_temp.eval()
            
            model_weight_temp.load_state_dict(model_weight_train.state_dict(), strict=False)
            model_weight_temp.eval()

            epoch_loss_3d_valid = None
            epoch_loss_3d_depth_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            epoch_loss_3d_vel = 0
            N = 0
            iteration = 0
            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d, batch_valid, _ in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_valid = torch.from_numpy(batch_valid.astype('float32'))

                    ##### apply test-time-augmentation (following Videopose3d)
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, :, 0] *= -1
                    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                    ##### convert size
                    inputs_3d_p = inputs_3d
                    inputs_2d, inputs_3d, valid_frame = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p, inputs_valid)
                    inputs_2d_flip, _, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p, inputs_valid)

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()
                    inputs_3d[:, :, 14] = 0

                    bs = 4
                    total_batch = (inputs_3d.shape[0] + bs - 1) // bs

                    for batch_cnt in range(total_batch):

                        if (batch_cnt + 1) * bs > inputs_3d.shape[0]:
                            inputs_2d_single = inputs_2d[batch_cnt * bs:]
                            inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:]
                            inputs_3d_single = inputs_3d[batch_cnt * bs:]
                            valid_frame_single = valid_frame[batch_cnt * bs:]
                        else:
                            inputs_2d_single = inputs_2d[batch_cnt * bs:(batch_cnt + 1) * bs]
                            inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:(batch_cnt + 1) * bs]
                            inputs_3d_single = inputs_3d[batch_cnt * bs:(batch_cnt + 1) * bs]
                            valid_frame_single = valid_frame[batch_cnt * bs:(batch_cnt + 1) * bs]


                        predicted_3d_pos_single = model_pos_test_temp(inputs_2d_single, inputs_3d_single,
                                                      input_2d_flip=inputs_2d_flip_single)  # b, t, h, f, j, c
                        b,t,h,f,j,c = predicted_3d_pos_single.shape
                        predicted_3d_pos_single = predicted_3d_pos_single.reshape(-1, j, c)
                        predicted_3d_pos_single = model_weight_temp(predicted_3d_pos_single)
                        predicted_3d_pos_single = predicted_3d_pos_single.reshape(b, t, h, f, j, c)
                        predicted_3d_pos_single[:, :, :, :, 14] = 0

                        error = mpjpe_diffusion_3dhp(predicted_3d_pos_single, inputs_3d_single, valid_frame_single.type(torch.bool))

                        if iteration == 0:
                            epoch_loss_3d_valid = inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error.clone()
                        else:
                            epoch_loss_3d_valid += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error.clone()

                        N += inputs_3d_single.shape[0] * inputs_3d_single.shape[1]

                        iteration += 1

                        if quickdebug:
                            if N == inputs_3d_single.shape[0] * inputs_3d_single.shape[1]:
                                break

                    if quickdebug:
                        if N == inputs_3d_single.shape[0] * inputs_3d_single.shape[1]:
                            break


                losses_3d_valid.append(epoch_loss_3d_valid / N)


        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 2d_pos_trainn %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_pos_train[-1] * 1000,
                losses_2d_pos_train[-1] * 1000
            ))

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            f = open(log_path, mode='a')
            f.write('[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 2d_pos_train %f\n' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_pos_train[-1] * 1000,
                losses_2d_pos_train[-1] * 1000
            ))
            f.close()

        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 2d_pos_train %f 3d_pos_valid %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1],
                losses_3d_pos_train[-1],
                losses_2d_pos_train[-1],
                losses_3d_valid[-1][0]
            ))

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            f = open(log_path, mode='a')
            f.write('[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 2d_pos_train %f 3d_pos_valid %f\n' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1],
                losses_3d_pos_train[-1],
                losses_2d_pos_train[-1],
                losses_3d_valid[-1][0]
            ))
            f.close()

            if not args.nolog:
                #writer.add_scalar("Loss/3d training eval loss", losses_3d_train_eval[-1] * 1000, epoch+1)
                writer.add_scalar("Loss/3d validation loss", losses_3d_valid[-1] * 1000, epoch+1)
        if not args.nolog:
            writer.add_scalar("Loss/3d training loss", losses_3d_train[-1] * 1000, epoch+1)
            writer.add_scalar("Parameters/learing rate", lr, epoch+1)
            writer.add_scalar('Parameters/training time per epoch', elapsed, epoch+1)
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_weight': model_weight_train.state_dict(),
                # 'min_loss': min_loss
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
        # min_loss = 41.65
        if losses_3d_valid[-1][0] < min_loss:
            min_loss = losses_3d_valid[-1]
            best_epoch = epoch
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_weight': model_weight_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, best_chk_path)

            f = open(log_path, mode='a')
            f.write('best epoch\n')
            f.close()

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')
# Training end

# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False, newmodel=None):
    epoch_loss_3d_pos = torch.zeros(args.sampling_timesteps).cuda()
    epoch_loss_3d_pos_mean = torch.zeros(args.sampling_timesteps).cuda()

    with torch.no_grad():
        if newmodel is not None:
            print('Loading comparison model')
            model_eval = newmodel
            chk_file_path = '/mnt/data3/home/zjl/workspace/3dpose/PoseFormer/checkpoint/train_pf_00/epoch_60.bin'
            print('Loading evaluate checkpoint of comparison model', chk_file_path)
            checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
            model_eval.load_state_dict(checkpoint['model_pos'], strict=False)
            model_eval.eval()
        else:
            model_eval = model_pos
            model_eval1 = model_weight
            if not use_trajectory_model:
                # load best checkpoint
                if args.evaluate == '':
                    chk_file_path = os.path.join(args.checkpoint, 'best_epoch.bin')
                    print('Loading best checkpoint', chk_file_path)
                elif args.evaluate != '':
                    chk_file_path = os.path.join(args.checkpoint, args.evaluate)
                    print('Loading evaluate checkpoint', chk_file_path)
                checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
                print('This model was trained for {} epochs'.format(checkpoint['epoch']))
                # model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
                model_eval.load_state_dict(checkpoint['model_pos'])
                model_eval.eval()
                
                model_eval1.load_state_dict(checkpoint['model_weight'])
                model_eval1.eval()
        # else:
            # model_traj.eval()
        N = 0
        iteration = 0
        data_inference_all = {}
        data_inference_mean = {}
        data_inference_h_min = {}
        data_inference_joint_min = {}
        data_inference_reproj_min = {}

        cam_1 = torch.tensor([7.32506, 7.32506, -0.0322884, 0.0929296, 0, 0, 0, 0, 0])
        cam_data_1 = [2048, 2048, 10, 10] #width, height, sensorSize_x, sensorSize_y
        cam_2 = torch.tensor([8.770747185, 8.770747185, -0.104908645, 0.104899704, 0, 0, 0, 0, 0])
        cam_data_2 = [1920, 1080, 10, 5.625]  # width, height, sensorSize_x, sensorSize_y
        cam_1 = cam_mm_to_pix(cam_1, cam_data_1)
        cam_2 = cam_mm_to_pix(cam_2, cam_data_2)
        
        quickdebug=args.debug
        for _, batch, batch_2d, batch_valid, keys in test_generator.next_epoch():
            # if keys != "TS5":
            #     continue

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_valid = torch.from_numpy(batch_valid.astype('float32'))

            _, f_sz, j_sz, c_sz = inputs_3d.shape
            data_inference_all[keys] = np.zeros((args.sampling_timesteps, args.num_proposals, f_sz, j_sz, c_sz))
            data_inference_mean[keys] = np.zeros((args.sampling_timesteps, f_sz, j_sz, c_sz))
            data_inference_h_min[keys] = np.zeros((args.sampling_timesteps, f_sz, j_sz, c_sz))
            data_inference_joint_min[keys] = np.zeros((args.sampling_timesteps, f_sz, j_sz, c_sz))
            data_inference_reproj_min[keys] = np.zeros((args.sampling_timesteps, f_sz, j_sz, c_sz))

            print(keys)

            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            inputs_3d_p = inputs_3d
            if newmodel is not None:
                def eval_data_prepare_pf(receptive_field, inputs_2d, inputs_3d):
                    inputs_2d_p = torch.squeeze(inputs_2d)
                    inputs_3d_p = inputs_3d.permute(1,0,2,3)
                    padding = int(receptive_field//2)
                    inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
                    inputs_2d_p = F.pad(inputs_2d_p, (padding,padding), mode='replicate')
                    inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
                    out_num = inputs_2d_p.shape[0] - receptive_field + 1
                    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
                    for i in range(out_num):
                        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
                    return eval_input_2d, inputs_3d_p
                
                inputs_2d, inputs_3d = eval_data_prepare_pf(81, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare_pf(81, inputs_2d_flip, inputs_3d_p)
            else:
                inputs_2d, inputs_3d, valid_frame = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p, inputs_valid)
                inputs_2d_flip, _, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p, inputs_valid)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()


            bs = 2
            total_batch = (inputs_3d.shape[0] + bs - 1) // bs

            for batch_cnt in range(total_batch):

                if (batch_cnt + 1) * bs > inputs_3d.shape[0]:
                    inputs_2d_single = inputs_2d[batch_cnt * bs:]
                    inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:]
                    inputs_3d_single = inputs_3d[batch_cnt * bs:]
                    valid_frame_single = valid_frame[batch_cnt * bs:]
                else:
                    inputs_2d_single = inputs_2d[batch_cnt * bs:(batch_cnt+1) * bs]
                    inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:(batch_cnt+1) * bs]
                    inputs_3d_single = inputs_3d[batch_cnt * bs:(batch_cnt+1) * bs]
                    valid_frame_single = valid_frame[batch_cnt * bs:(batch_cnt + 1) * bs]

                traj = inputs_3d_single[:, :, 14:15].clone()
                inputs_3d_single[:, :, 14] = 0

                predicted_3d_pos_single = model_eval(inputs_2d_single, inputs_3d_single, input_2d_flip=inputs_2d_flip_single) #b, t, h, f, j, c
                b,t,h,f,j,c = predicted_3d_pos_single.shape
                predicted_3d_pos_single = predicted_3d_pos_single.reshape(-1, j, c)
                predicted_3d_pos_single = model_eval1(predicted_3d_pos_single)
                predicted_3d_pos_single = predicted_3d_pos_single.reshape(b, t, h, f, j, c)
                
                predicted_3d_pos_single[:, :, :, :, 14] = 0

                mean_pose = torch.mean(predicted_3d_pos_single, dim=2, keepdim=False)

                # inputs_traj_single_all = traj.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
                # predicted_3d_pos_abs_single = predicted_3d_pos_single + inputs_traj_single_all
                # #predicted_3d_pos_abs_single = predicted_3d_pos_abs_single/1000
                # predicted_3d_pos_abs_single = predicted_3d_pos_abs_single.reshape(b * t * h * f, j_sz, c_sz)
                # if keys == "TS5" or keys == "TS6":
                #     cam = cam_2.clone()
                #     cam_data = cam_data_2.copy()
                #     reproject_func = project_to_2d
                # else:
                #     cam = cam_1.clone()
                #     cam_data = cam_data_1.copy()
                #     reproject_func = project_to_2d_linear

                # cam_single_all = cam.unsqueeze(0).repeat(b * t * h * f, 1).cuda()
                # reproj_2d = reproject_func(predicted_3d_pos_abs_single, cam_single_all)
                # reproj_2d = reproj_2d.reshape(b, t, h, f, j_sz, 2)

                # cam_single_gt = cam.unsqueeze(0).repeat(b * f, 1).cuda()
                # input_3d_reproj = inputs_3d_single + traj
                # # input_3d_reproj = input_3d_reproj / 1000
                # input_3d_reproj = input_3d_reproj.reshape(b * f, 17, 3)
                # reproj_2d_gt = reproject_func(input_3d_reproj, cam_single_gt)
                # reproj_2d_gt = reproj_2d_gt.reshape(b, f, 17, 2)

                # target_2d = torch.from_numpy(image_coordinates(inputs_2d_single[..., :2].cpu().numpy(), w=cam_data[0], h=cam_data[1])).cuda()
                # target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
                # errors_2d = torch.norm(reproj_2d - target_2d, dim=len(target_2d.shape) - 1) # b, t, h, f, n
                # reproj_min_indices = torch.min(errors_2d, dim=2, keepdim=True).indices  # b,t,1,f,n
                # reproj_min_indices = reproj_min_indices.unsqueeze(-1).repeat(1, 1, 1, 1, 1, c_sz)
                # reproj_min_pose = torch.gather(predicted_3d_pos_single, 2, reproj_min_indices).squeeze(2)

            
                if return_predictions:
                    return predicted_3d_pos_single.squeeze().cpu().numpy()

                error = mpjpe_diffusion_3dhp(predicted_3d_pos_single, inputs_3d_single, valid_frame_single.type(torch.bool))
                error_mean = mpjpe_diffusion_3dhp(predicted_3d_pos_single, inputs_3d_single, valid_frame_single.type(torch.bool), mean_pos=True)

                epoch_loss_3d_pos += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error.clone()
                epoch_loss_3d_pos_mean += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error_mean.clone()

                N += inputs_3d_single.shape[0] * inputs_3d_single.shape[1]


            data_inference_mean = pose_post_process(mean_pose, data_inference_mean, keys, receptive_field)
            

            log_path = os.path.join(args.checkpoint, '3dhp_test_log_H%d_K%d.txt' %(args.num_proposals, args.sampling_timesteps))
            f = open(log_path, mode='a')
            if keys is None:
                print('----------')
            else:
                print('----'+keys+'----')
                f.write('----'+keys+'----\n')

            e1 = (epoch_loss_3d_pos / N)
            e1_mean = (epoch_loss_3d_pos_mean / N)

            print('Test time augmentation:', test_generator.augment_enabled())
            for ii in range(e1.shape[0]):
                print('step %d : Protocol #1 Error (MPJPE) P_Best:' % ii, e1[ii].item(), 'mm')
                f.write('step %d : Protocol #1 Error (MPJPE) P_Best: %f mm\n' % (ii, e1[ii].item()))
                print('step %d : Protocol #1 Error (MPJPE) P_Agg:' % ii, e1_mean[ii].item(), 'mm')
                f.write('step %d : Protocol #1 Error (MPJPE) P_Agg: %f mm\n' % (ii, e1_mean[ii].item()))

            print('----------')
            f.write('----------\n')

            f.close()

            if quickdebug:
                break

        # mat_path_all = os.path.join(args.checkpoint, 'inference_data_all.mat')
        # scio.savemat(mat_path_all, data_inference_all)
        mat_path_mean = os.path.join(args.checkpoint, 'inference_data.mat')
        scio.savemat(mat_path_mean, data_inference_mean)
        
    return e1, e1_mean

if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator_Seq(None, [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    if args.compare:
        from common.model_poseformer import PoseTransformer
        model_pf = PoseTransformer(num_frame=81, num_joints=17, in_chans=2, num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,drop_path_rate=0.1)
        if torch.cuda.is_available():
            model_pf = nn.DataParallel(model_pf)
            model_pf = model_pf.cuda()
        prediction_pf = evaluate(gen, newmodel=model_pf, return_predictions=True)
        
        # ### reshape prediction_pf as ground truth
        # if ground_truth.shape[0] / receptive_field > ground_truth.shape[0] // receptive_field: 
        #     batch_num = (ground_truth.shape[0] // receptive_field) +1
        #     prediction_pf_2 = np.empty_like(ground_truth)
        #     for i in range(batch_num-1):
        #         prediction_pf_2[i*receptive_field:(i+1)*receptive_field,:,:] = prediction_pf[i,:,:,:]
        #     left_frames = ground_truth.shape[0] - (batch_num-1)*receptive_field
        #     prediction_pf_2[-left_frames:,:,:] = prediction_pf[-1,-left_frames:,:,:]
        #     prediction_pf = prediction_pf_2
        # elif ground_truth.shape[0] / receptive_field == ground_truth.shape[0] // receptive_field:
        #     prediction_pf.reshape(ground_truth.shape[0], 17, 3)

    # if model_traj is not None and ground_truth is None:
    #     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
    #     prediction += prediction_traj
    ### reshape prediction as ground truth
    if ground_truth.shape[0] / receptive_field > ground_truth.shape[0] // receptive_field: 
        batch_num = (ground_truth.shape[0] // receptive_field) +1
        prediction2 = np.empty_like(ground_truth)
        for i in range(batch_num-1):
            prediction2[i*receptive_field:(i+1)*receptive_field,:,:] = prediction[i,:,:,:]
        left_frames = ground_truth.shape[0] - (batch_num-1)*receptive_field
        prediction2[-left_frames:,:,:] = prediction[-1,-left_frames:,:,:]
        prediction = prediction2
    elif ground_truth.shape[0] / receptive_field == ground_truth.shape[0] // receptive_field:
        prediction.reshape(ground_truth.shape[0], 17, 3)

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory
            if args.compare:
                prediction_pf += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            if args.compare:
                prediction_pf = camera_to_world(prediction_pf, R=cam['orientation'], t=cam['translation'])
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            if args.compare:
                prediction_pf = camera_to_world(prediction_pf, R=rot, t=0)
                prediction_pf[:, :, 2] -= np.min(prediction_pf[:, :, 2])
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        
        if args.compare:
            anim_output = {'PoseFormer': prediction_pf}
            anim_output['Ours'] = prediction
            # print(prediction_pf.shape, prediction.shape)
        else:
            # anim_output = {'Reconstruction': prediction}
            anim_output = {'Reconstruction': ground_truth + np.random.normal(loc=0.0, scale=0.1, size=[ground_truth.shape[0], 17, 3])}
        
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from common.visualization import render_animation
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                        dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                        limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                        input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                        input_video_skip=args.viz_skip)

else:
    print('Evaluating...')
    all_actions = {}
    all_actions_flatten = []
    all_actions_by_subject = {}

    def run_evaluation_all_actions(actions, action_filter=None):
        errors_p1 = []
        errors_p1_mean = []


        #poses_act, poses_2d_act = fetch_actions(actions)
        gen = UnchunkedGenerator_Seq(None, out_poses_3d_test, out_poses_2d_test,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right, valid_frame=valid_frame)
        #e1, e2, e3, ev = evaluate(gen)
        e1, e1_mean = evaluate(gen)

        # joints_errs_list.append(joints_errs)

        errors_p1.append(e1)
        errors_p1_mean.append(e1_mean)


    if not args.by_subject:
        #run_evaluation(all_actions, action_filter)
        run_evaluation_all_actions(all_actions_flatten, action_filter)

if not args.nolog:
    writer.close()