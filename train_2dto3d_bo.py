#!/usr/bin/env python3
"""
train_pose2mesh_joints_bo.py

Uses Bayesian Optimization (Optuna) to find optimal loss weights for 
Pose2Mesh training with geometric constraints.

For each trial:
- Trains for a limited number of epochs with suggested hyperparameters
- Evaluates on validation set
- Visualizes 5 test samples with 2D and 3D predictions
"""

import os
import random
import argparse
import h5py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import optuna
from optuna.trial import TrialState
import json
import pickle

# ----------------------------------------------------------------------
# 1. H36MDataset (same as original)
# ----------------------------------------------------------------------
class H36MDataset(Dataset):
    def __init__(self, data_path, split="train", image_size=(1000, 1000)):
        self.data_path = data_path
        self.split = split
        self.image_size = image_size  # (H, W)

        annot_dir = os.path.join(data_path, "annot")
        if split == "train":
            img_list_file = os.path.join(annot_dir, "train_images.txt")
            h5_file = os.path.join(annot_dir, "train.h5")
        else:
            img_list_file = os.path.join(annot_dir, "valid_images.txt")
            h5_file = os.path.join(annot_dir, "valid.h5")

        # Load image paths
        with open(img_list_file, "r") as f:
            self.image_names = [line.strip() for line in f]

        # Load raw 2D and 3D from HDF5
        with h5py.File(h5_file, "r") as f:
            raw_2d = f["part"][:]
            raw_3d = f["S"][:]

        # Transpose if needed → (N,17,2)
        if raw_2d.ndim == 3 and raw_2d.shape[1] == 2 and raw_2d.shape[2] == 17:
            joints_2d = raw_2d.transpose(0, 2, 1).astype(np.float32)
        else:
            joints_2d = raw_2d.astype(np.float32)

        # Transpose if needed → (N,17,3)
        if raw_3d.ndim == 3 and raw_3d.shape[1] == 3 and raw_3d.shape[2] == 17:
            joints_3d = raw_3d.transpose(0, 2, 1).astype(np.float32)
        else:
            joints_3d = raw_3d.astype(np.float32)

        assert (
            len(self.image_names) == joints_2d.shape[0] == joints_3d.shape[0]
        ), f"Mismatch: images {len(self.image_names)}, 2D {joints_2d.shape[0]}, 3D {joints_3d.shape[0]}"

        self.joints_2d = joints_2d  # (N,17,2) pixels
        self.joints_3d = joints_3d  # (N,17,3) meters

    def __len__(self):
        return len(self.joints_2d)

    def __getitem__(self, idx):
        k2d = self.joints_2d[idx].copy()  # (17,2) in pixels
        k3d = self.joints_3d[idx].copy()  # (17,3) in meters

        H, W = self.image_size

        # Normalize 2D
        x = k2d[:, 0]
        y = k2d[:, 1]
        x_n = (x - (W / 2.0)) / (W / 2.0)
        y_n = - (y - (H / 2.0)) / (H / 2.0)
        keypoints2d = np.stack([x_n, y_n], axis=1).astype(np.float32)

        # Root-center 3D
        root = k3d[0:1, :]
        centered_3d = (k3d - root).astype(np.float32)

        return torch.from_numpy(keypoints2d), torch.from_numpy(centered_3d)

# ----------------------------------------------------------------------
# 2. Skeleton connectivity & adjacency normalization
# ----------------------------------------------------------------------
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),        # Right leg
    (0, 4), (4, 5), (5, 6),        # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine → head
    (11, 12), (12, 13),            # Left arm
    (14, 15), (15, 16),            # Right arm
    (11, 8), (8, 14)               # Shoulders → thorax
]

def build_adjacency(num_joints=17, skeleton=H36M_SKELETON):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for (u, v) in skeleton:
        A[u, v] = 1.0
        A[v, u] = 1.0
    for i in range(num_joints):
        A[i, i] = 1.0
    return A

def normalize_adjacency(A):
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=(D>0))
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat
    return torch.from_numpy(A_norm.astype(np.float32))

# ----------------------------------------------------------------------
# 3. GCN model (same as original)
# ----------------------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, A_norm):
        x_agg = torch.einsum("ij,bjk->bik", A_norm, x)
        out = self.fc(x_agg)
        return out

class Pose2MeshJoints(nn.Module):
    def __init__(self, A_norm, hidden_dims=[64, 128, 128, 64, 32]):
        super(Pose2MeshJoints, self).__init__()
        self.A_norm = A_norm

        # Input MLP: 2 → 64
        self.lin_in = nn.Linear(2, hidden_dims[0])

        # GraphConv blocks
        self.gcns = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gcns.append(GraphConv(hidden_dims[i], hidden_dims[i+1]))

        # BatchNorm for each hidden block
        self.bns = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_dims[1:]])

        # Final GraphConv: 32 → 3
        self.gc_out = GraphConv(hidden_dims[-1], 3)

        self.relu = nn.ReLU()

    def forward(self, x2d):
        B, N, _ = x2d.shape

        # Project 2D → 64
        h = self.lin_in(x2d)
        h = self.relu(h)

        # GCN blocks with BN + ReLU
        for idx, gcn in enumerate(self.gcns):
            h = gcn(h, self.A_norm)
            bn = self.bns[idx]
            h = bn(h.transpose(1, 2)).transpose(1, 2)
            h = self.relu(h)

        # Final 3D prediction
        x3d = self.gc_out(h, self.A_norm)
        return x3d

# ----------------------------------------------------------------------
# 4. Geometric constraint losses (with configurable weights)
# ----------------------------------------------------------------------
def compute_angle_loss(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    v1_norm = F.normalize(v1, dim=-1)
    v2_norm = F.normalize(v2, dim=-1)
    cos_angle = torch.sum(v1_norm * v2_norm, dim=-1)
    return cos_angle

def compute_plane_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = torch.cross(v1, v2, dim=-1)
    return F.normalize(normal, dim=-1)

def compute_bone_length_ratios(pred3d):
    # Joint indices
    pelvis, r_hip, r_knee, r_foot = 0, 1, 2, 3
    l_hip, l_knee, l_foot = 4, 5, 6
    spine, thorax, neck, head = 7, 8, 9, 10
    l_shoulder, l_elbow, l_wrist = 11, 12, 13
    r_shoulder, r_elbow, r_wrist = 14, 15, 16
    
    # Reference ratios (normalized by upper arm length as base = 1.0)
    ref_ratios = {
        'upper_arm': 1.0,
        'forearm': 0.787,
        'thigh': 1.671,
        'shank': 1.098,
        'shoulder_width': 1.220,
        'hip_width': 0.957,
        'torso': 1.040,
    }
    
    # Compute actual bone lengths
    l_upper_arm = torch.norm(pred3d[:, l_elbow] - pred3d[:, l_shoulder], dim=-1)
    r_upper_arm = torch.norm(pred3d[:, r_elbow] - pred3d[:, r_shoulder], dim=-1)
    l_forearm = torch.norm(pred3d[:, l_wrist] - pred3d[:, l_elbow], dim=-1)
    r_forearm = torch.norm(pred3d[:, r_wrist] - pred3d[:, r_elbow], dim=-1)
    l_thigh = torch.norm(pred3d[:, l_knee] - pred3d[:, l_hip], dim=-1)
    r_thigh = torch.norm(pred3d[:, r_knee] - pred3d[:, r_hip], dim=-1)
    l_shank = torch.norm(pred3d[:, l_foot] - pred3d[:, l_knee], dim=-1)
    r_shank = torch.norm(pred3d[:, r_foot] - pred3d[:, r_knee], dim=-1)
    shoulder_width = torch.norm(pred3d[:, r_shoulder] - pred3d[:, l_shoulder], dim=-1)
    hip_width = torch.norm(pred3d[:, r_hip] - pred3d[:, l_hip], dim=-1)
    
    # For torso
    shoulder_center = (pred3d[:, l_shoulder] + pred3d[:, r_shoulder]) / 2
    hip_center = (pred3d[:, l_hip] + pred3d[:, r_hip]) / 2
    torso_height = torch.norm(hip_center - shoulder_center, dim=-1)
    
    # Use average upper arm length as reference
    ref_bone = (l_upper_arm + r_upper_arm) / 2
    
    # Compute ratios and losses
    ratio_losses = []
    
    # Forearm ratios
    l_forearm_ratio = l_forearm / (l_upper_arm + 1e-6)
    r_forearm_ratio = r_forearm / (r_upper_arm + 1e-6)
    ratio_losses.append((l_forearm_ratio - ref_ratios['forearm']) ** 2)
    ratio_losses.append((r_forearm_ratio - ref_ratios['forearm']) ** 2)
    
    # Thigh ratios
    l_thigh_ratio = l_thigh / (ref_bone + 1e-6)
    r_thigh_ratio = r_thigh / (ref_bone + 1e-6)
    ratio_losses.append((l_thigh_ratio - ref_ratios['thigh']) ** 2)
    ratio_losses.append((r_thigh_ratio - ref_ratios['thigh']) ** 2)
    
    # Shank ratios
    l_shank_ratio = l_shank / (ref_bone + 1e-6)
    r_shank_ratio = r_shank / (ref_bone + 1e-6)
    ratio_losses.append((l_shank_ratio - ref_ratios['shank']) ** 2)
    ratio_losses.append((r_shank_ratio - ref_ratios['shank']) ** 2)
    
    # Width ratios
    shoulder_ratio = shoulder_width / (ref_bone + 1e-6)
    hip_ratio = hip_width / (ref_bone + 1e-6)
    ratio_losses.append((shoulder_ratio - ref_ratios['shoulder_width']) ** 2)
    ratio_losses.append((hip_ratio - ref_ratios['hip_width']) ** 2)
    
    # Torso ratio
    torso_ratio = torso_height / (ref_bone + 1e-6)
    ratio_losses.append((torso_ratio - ref_ratios['torso']) ** 2)
    
    return torch.mean(torch.stack(ratio_losses, dim=1))

def compute_coronal_plane_normal(pred3d):
    ls = pred3d[:, 11]   # left shoulder
    rs = pred3d[:, 14]   # right shoulder
    pel = pred3d[:, 0]   # pelvis
    shoulder_center = (ls + rs) / 2.0

    v1 = rs - ls
    v2 = pel - shoulder_center

    n = torch.cross(v1, v2, dim=-1)
    return F.normalize(n, dim=-1)

def compute_geometric_constraints(pred3d, epoch, weights_dict):
    """
    pred3d: (B,17,3)
    epoch: int
    weights_dict: dictionary containing weight values for each loss term
    Returns losses dict and weights dict.
    """
    B = pred3d.shape[0]
    losses, weights = {}, {}

    # only start at epoch 51
    if epoch <= 20:
        return losses, weights
    ramp = min((epoch - 20) / 20.0, 1.0)

    # indices
    pel, r_hip, r_knee, r_foot = 0, 1, 2, 3
    l_hip, l_knee, l_foot = 4, 5, 6
    spine, thorax, neck, head = 7, 8, 9, 10
    ls, le, lw = 11, 12, 13
    rs, re, rw = 14, 15, 16

    # reusable angle-cos
    def cos_ang(a,b,c):
        return compute_angle_loss(a,b,c)

    # --- Strong constraints ---
    # 1) hip chain straight
    hip_cos = cos_ang(pred3d[:, l_hip], pred3d[:, pel], pred3d[:, r_hip])
    loss = torch.mean(F.relu(hip_cos))
    losses['strong_hip_straight'] = loss
    weights['strong_hip_straight'] = weights_dict['strong_hip_straight'] * ramp

    # 2) shoulder chain >90°
    sh_cos = cos_ang(pred3d[:, ls], pred3d[:, thorax], pred3d[:, rs])
    loss = torch.mean(F.relu(sh_cos))
    losses['strong_shoulder_angle'] = loss
    weights['strong_shoulder_angle'] = weights_dict['strong_shoulder_angle'] * ramp

    # 3) neck-thorax-spine >90°
    nts_cos = cos_ang(pred3d[:, neck], pred3d[:, thorax], pred3d[:, spine])
    loss = torch.mean(F.relu(nts_cos))
    losses['strong_neck_thorax_spine'] = loss
    weights['strong_neck_thorax_spine'] = weights_dict['strong_neck_thorax_spine'] * ramp

    # 4) head-neck-thorax >90°
    hnt_cos = cos_ang(pred3d[:, head], pred3d[:, neck], pred3d[:, thorax])
    loss = torch.mean(F.relu(hnt_cos))
    losses['strong_head_neck_thorax'] = loss
    weights['strong_head_neck_thorax'] = weights_dict['strong_head_neck_thorax'] * ramp

    # 5) never curve backward in thorax-spine-pelvis
    front_n = compute_coronal_plane_normal(pred3d)
    v1 = pred3d[:, thorax] - pred3d[:, spine]
    v2 = pred3d[:, pel]    - pred3d[:, spine]
    bis = F.normalize(v1 + v2, dim=-1)
    loss = torch.mean(F.relu(-torch.sum(bis * front_n, dim=-1)))
    losses['strong_thorax_spine_pelvis'] = loss
    weights['strong_thorax_spine_pelvis'] = weights_dict['strong_thorax_spine_pelvis'] * ramp

    # 6) hip-knee-foot never curve in front
    v1 = pred3d[:, l_hip] - pred3d[:, l_knee]
    v2 = pred3d[:, l_foot]- pred3d[:, l_knee]
    bis_l = F.normalize(v1 + v2, dim=-1)
    loss_l = torch.mean(F.relu(torch.sum(bis_l * front_n, dim=-1)))
    v1 = pred3d[:, r_hip] - pred3d[:, r_knee]
    v2 = pred3d[:, r_foot]- pred3d[:, r_knee]
    bis_r = F.normalize(v1 + v2, dim=-1)
    loss_r = torch.mean(F.relu(torch.sum(bis_r * front_n, dim=-1)))
    losses['strong_hip_knee_foot'] = 0.5 * (loss_l + loss_r)
    weights['strong_hip_knee_foot'] = weights_dict['strong_hip_knee_foot'] * ramp

    # --- Medium constraints ---
    # shoulder straight (weakly)
    loss = -torch.mean(cos_ang(pred3d[:, ls], pred3d[:, thorax], pred3d[:, rs]))
    losses['med_shoulder_straight'] = loss
    weights['med_shoulder_straight'] = weights_dict['med_shoulder_straight'] * ramp

    # neck-thorax-spine straight
    loss = -torch.mean(cos_ang(pred3d[:, neck], pred3d[:, thorax], pred3d[:, spine]))
    losses['med_neck_spine_straight'] = loss
    weights['med_neck_spine_straight'] = weights_dict['med_neck_spine_straight'] * ramp

    # bone-length ratios
    bone_loss = compute_bone_length_ratios(pred3d)
    losses['med_bone_ratios'] = bone_loss
    weights['med_bone_ratios'] = weights_dict['med_bone_ratios'] * ramp

    # --- Weak constraints ---
    # vertical spine-hip-pelvis / pelvis-hip-knee
    vert = torch.tensor([0, -1, 0], device=pred3d.device).float()
    ph = pred3d[:, l_hip] - pred3d[:, pel]
    sp = pred3d[:, spine] - pred3d[:, pel]
    w1 = 1 - torch.abs(F.cosine_similarity(ph, vert.unsqueeze(0), dim=-1))
    w2 = 1 - torch.abs(F.cosine_similarity(sp, vert.unsqueeze(0), dim=-1))
    losses['weak_spine_vertical'] = torch.mean(w1 + w2)
    weights['weak_spine_vertical'] = weights_dict['weak_spine_vertical'] * ramp

    hk = pred3d[:, l_knee] - pred3d[:, l_hip]
    w3 = 1 - torch.abs(F.cosine_similarity(hk, vert.unsqueeze(0), dim=-1))
    hk2= pred3d[:, r_knee] - pred3d[:, r_hip]
    w4 = 1 - torch.abs(F.cosine_similarity(hk2, vert.unsqueeze(0), dim=-1))
    losses['weak_hip_knee_vertical'] = torch.mean(w3 + w4)
    weights['weak_hip_knee_vertical'] = weights_dict['weak_hip_knee_vertical'] * ramp

    # coronal-orientation
    ws_l = pred3d[:, lw] - ((pred3d[:, ls] + pred3d[:, rs]) / 2)
    ws_r = pred3d[:, rw] - ((pred3d[:, ls] + pred3d[:, rs]) / 2)
    loss = torch.mean(F.relu(-torch.sum(ws_l * front_n, dim=-1)))
    loss += torch.mean(F.relu(-torch.sum(ws_r * front_n, dim=-1)))
    ks_l = pred3d[:, l_knee] - ((pred3d[:, l_hip] + pred3d[:, l_foot]) / 2)
    ks_r = pred3d[:, r_knee] - ((pred3d[:, r_hip] + pred3d[:, r_foot]) / 2)
    loss += torch.mean(F.relu(-torch.sum(ks_l * front_n, dim=-1)))
    loss += torch.mean(F.relu(-torch.sum(ks_r * front_n, dim=-1)))
    hs = pred3d[:, head] - ((pred3d[:, neck] + pred3d[:, thorax]) / 2)
    loss += torch.mean(F.relu(-torch.sum(hs  * front_n, dim=-1)))
    losses['weak_coronal_orientation'] = loss / 5.0
    weights['weak_coronal_orientation'] = weights_dict['weak_coronal_orientation'] * ramp

    return losses, weights

# ----------------------------------------------------------------------
# 5. Plotting utilities (updated for BO visualization)
# ----------------------------------------------------------------------
def build_image_lookup(root_dir):
    lookup = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                lookup[fname] = os.path.join(dirpath, fname)
    return lookup

def plot_2d_on_axis(ax, image, keypoints, skeleton, radius=1, color=(0,1,0), linewidth=2):
    ax.imshow(image)
    ax.axis("off")
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
    ax.scatter(xs, ys, c="r", s=(radius*5)**2, edgecolors="white", linewidths=0.5)
    for (p, c) in skeleton:
        ax.plot(
            [keypoints[p,0], keypoints[c,0]],
            [keypoints[p,1], keypoints[c,1]],
            c=color, linewidth=linewidth
        )

def rotate_x_minus90(joints_3d):
    x = joints_3d[:,0]; y = joints_3d[:,1]; z = joints_3d[:,2]
    return np.stack([ x, z, -y ], axis=1)

def rotate_combined(joints_3d):
    return rotate_x_minus90(joints_3d)

def plot_3d_on_axis(ax3d, joints_3d_gt, joints_3d_pred, skeleton, title=None):
    """
    Draw GT 3D skeleton (blue) and predicted 3D skeleton (red) on same axis.
    """
    # Process GT
    pts_gt = joints_3d_gt.copy()
    if np.max(np.abs(pts_gt)) < 10:  # meters → mm
        pts_gt = pts_gt * 1000.0
    centered_gt = pts_gt - pts_gt[0]
    rotated_gt = rotate_combined(centered_gt)

    # Process Pred
    pts_pred = joints_3d_pred.copy()
    if np.max(np.abs(pts_pred)) < 10:  # meters → mm
        pts_pred = pts_pred * 1000.0
    centered_pred = pts_pred - pts_pred[0]
    rotated_pred = rotate_combined(centered_pred)

    # Plot GT in blue
    ax3d.scatter(rotated_gt[:,0], rotated_gt[:,1], rotated_gt[:,2], c="b", s=30, label="GT")
    for (p, c) in skeleton:
        ax3d.plot(
            [rotated_gt[p,0], rotated_gt[c,0]],
            [rotated_gt[p,1], rotated_gt[c,1]],
            [rotated_gt[p,2], rotated_gt[c,2]],
            c="b", linewidth=2
        )

    # Plot Pred in red
    ax3d.scatter(rotated_pred[:,0], rotated_pred[:,1], rotated_pred[:,2], c="r", s=30, marker="o", label="Pred")
    for (p, c) in skeleton:
        ax3d.plot(
            [rotated_pred[p,0], rotated_pred[c,0]],
            [rotated_pred[p,1], rotated_pred[c,1]],
            [rotated_pred[p,2], rotated_pred[c,2]],
            c="r", linewidth=2
        )

    ax3d.set_xlabel("X (mm)"); ax3d.set_ylabel("Y (mm)"); ax3d.set_zlabel("Z (mm)")
    if title:
        ax3d.set_title(title)
    
    # Set equal aspect ratio
    all_pts = np.vstack([rotated_gt, rotated_pred])
    xyz_min = np.min(all_pts, axis=0)
    xyz_max = np.max(all_pts, axis=0)
    max_range = np.max(xyz_max - xyz_min)/2.0
    mid_x = (xyz_max[0] + xyz_min[0])/2.0
    mid_y = (xyz_max[1] + xyz_min[1])/2.0
    mid_z = (xyz_max[2] + xyz_min[2])/2.0
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    ax3d.legend()

# ----------------------------------------------------------------------
# 6. MPJPE metric & training functions
# ----------------------------------------------------------------------
def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=2))

def train_one_epoch(model, dataloader, optimizer, device, epoch, weights_dict):
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_constraint_loss = 0.0
    
    for k2d, k3d in dataloader:
        k2d = k2d.to(device)
        k3d = k3d.to(device)

        optimizer.zero_grad()
        pred3d = model(k2d)
        
        # Main reconstruction loss
        recon_loss = F.mse_loss(pred3d, k3d)
        
        # Geometric constraints with custom weights
        constraint_losses, constraint_weights = compute_geometric_constraints(pred3d, epoch, weights_dict)
        constraint_loss = torch.tensor(0.0, device=device)
        for name, loss_val in constraint_losses.items():
            if name in constraint_weights:
                constraint_loss += constraint_weights[name] * loss_val
        
        # Total loss
        loss = recon_loss + constraint_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            batch_mpjpe = mpjpe(pred3d, k3d).item()

        total_loss += recon_loss.item() * k2d.shape[0]
        total_constraint_loss += constraint_loss.item() * k2d.shape[0]
        total_mpjpe += batch_mpjpe * k2d.shape[0]

    N = len(dataloader.dataset)
    return total_loss / N, total_mpjpe / N, total_constraint_loss / N

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    
    with torch.no_grad():
        for k2d, k3d in dataloader:
            k2d = k2d.to(device)
            k3d = k3d.to(device)
            pred3d = model(k2d)

            loss = F.mse_loss(pred3d, k3d)
            batch_mpjpe = mpjpe(pred3d, k3d).item()

            total_loss += loss.item() * k2d.shape[0]
            total_mpjpe += batch_mpjpe * k2d.shape[0]

    N = len(dataloader.dataset)
    return total_loss / N, total_mpjpe / N

# ----------------------------------------------------------------------
# 7. BO Visualization
# ----------------------------------------------------------------------
def visualize_bo_trial(model, dataset, image_lookup, device, trial_dir, num_samples=5):
    """
    Visualize predictions for a BO trial.
    """
    model.eval()
    os.makedirs(trial_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        k2d_norm, k3d_gt = dataset[idx]
        with torch.no_grad():
            pred3d = model(k2d_norm.unsqueeze(0).to(device))
        pred3d = pred3d.cpu().squeeze(0).numpy()
        gt3d = k3d_gt.numpy()

        # Raw 2D pixel coords
        raw_k2d = dataset.joints_2d[idx]

        # Find RGB image
        relpath = dataset.image_names[idx]
        basename = os.path.basename(relpath)
        img_path = image_lookup.get(basename)
        if img_path is None or not os.path.isfile(img_path):
            continue
        
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Create figure with 2D and 3D side by side
        fig = plt.figure(figsize=(16, 8))

        # Left: 2D overlay
        ax2d = fig.add_subplot(1, 2, 1)
        plot_2d_on_axis(ax2d, img_rgb, raw_k2d, H36M_SKELETON, radius=1, color=(0, 1, 0), linewidth=2)
        ax2d.set_title(f"2D Keypoints - Sample {i+1}")

        # Right: 3D GT and Pred
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
        plot_3d_on_axis(ax3d, gt3d, pred3d, H36M_SKELETON, title=f"3D Comparison - Sample {i+1}")

        plt.tight_layout()
        save_path = os.path.join(trial_dir, f"sample_{i+1}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

# ----------------------------------------------------------------------
# 8. Objective function for Optuna
# ----------------------------------------------------------------------
def objective(trial, args):
    """
    Objective function for Bayesian Optimization.
    """
    # Suggest hyperparameters (loss weights)
    weights_dict = {
        # Strong constraints
        'strong_hip_straight': trial.suggest_float('strong_hip_straight', 0.01, 1.0),
        'strong_shoulder_angle': trial.suggest_float('strong_shoulder_angle', 0.01, 1.0),
        'strong_neck_thorax_spine': trial.suggest_float('strong_neck_thorax_spine', 0.01, 1.0),
        'strong_head_neck_thorax': trial.suggest_float('strong_head_neck_thorax', 0.01, 1.0),
        'strong_thorax_spine_pelvis': trial.suggest_float('strong_thorax_spine_pelvis', 0.01, 1.0),
        'strong_hip_knee_foot': trial.suggest_float('strong_hip_knee_foot', 0.01, 1.0),
        # Medium constraints
        'med_shoulder_straight': trial.suggest_float('med_shoulder_straight', 0.01, 0.5),
        'med_neck_spine_straight': trial.suggest_float('med_neck_spine_straight', 0.01, 0.5),
        'med_bone_ratios': trial.suggest_float('med_bone_ratios', 0.01, 0.5),
        # Weak constraints
        'weak_spine_vertical': trial.suggest_float('weak_spine_vertical', 0.01, 0.3),
        'weak_hip_knee_vertical': trial.suggest_float('weak_hip_knee_vertical', 0.01, 0.3),
        'weak_coronal_orientation': trial.suggest_float('weak_coronal_orientation', 0.01, 0.3),
    }
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = build_adjacency(num_joints=17)
    A_norm = normalize_adjacency(A).to(device)
    
    # Create datasets
    train_dataset = H36MDataset(args.data_path, split="train", image_size=args.image_size)
    valid_dataset = H36MDataset(args.data_path, split="valid", image_size=args.image_size)
    
    # For speed, use a subset of data
    train_subset_size = min(5000, len(train_dataset))
    valid_subset_size = min(1000, len(valid_dataset))
    train_indices = random.sample(range(len(train_dataset)), train_subset_size)
    valid_indices = random.sample(range(len(valid_dataset)), valid_subset_size)
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_subset = torch.utils.data.Subset(valid_dataset, valid_indices)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = Pose2MeshJoints(A_norm, hidden_dims=args.hidden_dims).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)  # Scaled down milestones
    
    # Train for limited epochs
    bo_epochs = args.bo_epochs
    best_val_mpjpe = float("inf")
    
    for epoch in range(1, bo_epochs + 1):
        train_loss, train_mpjpe, train_constraint = train_one_epoch(
            model, train_loader, optimizer, device, epoch, weights_dict
        )
        val_loss, val_mpjpe = validate_one_epoch(model, valid_loader, device)
        
        scheduler.step()
        
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
        
        # Report intermediate values
        trial.report(val_mpjpe, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Save trial results and visualizations
    trial_dir = os.path.join(args.out_dir, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save weights
    with open(os.path.join(trial_dir, "weights.json"), "w") as f:
        json.dump(weights_dict, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(trial_dir, "model.pth"))
    
    # Visualize on test samples
    images_dir = os.path.join(args.data_path, "images")
    image_lookup = build_image_lookup(images_dir)
    visualize_bo_trial(model, valid_dataset, image_lookup, device, trial_dir, num_samples=5)
    
    # Save metrics
    metrics = {
        "best_val_mpjpe": best_val_mpjpe,
        "final_val_mpjpe": val_mpjpe,
        "final_val_mpjpe_mm": val_mpjpe * 1000,
    }
    with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return val_mpjpe

# ----------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------
def main(args):
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50),
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=None,
        catch=(Exception,),
    )
    
    # Save study results
    with open(os.path.join(args.out_dir, "study.pkl"), "wb") as f:
        pickle.dump(study, f)
    
    # Print results
    print("\n" + "="*60)
    print("Bayesian Optimization Complete!")
    print("="*60)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best MPJPE: {study.best_value * 1000:.2f} mm")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    
    # Save best parameters
    best_params_path = os.path.join(args.out_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # Create visualization of optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(args.out_dir, "optimization_history.html"))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(args.out_dir, "param_importances.html"))
    
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(os.path.join(args.out_dir, "param_slice.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str,
        default="/home/oliver/Documents/data/h36m",
        help="Root directory of Human3.6M"
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[1000, 1000],
        help="(H, W) used to normalize 2D keypoints"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--hidden_dims", type=int, nargs="+",
        default=[64, 128, 128, 64, 32],
        help="Hidden dims for GraphConv blocks"
    )
    parser.add_argument(
        "--bo_epochs", type=int, default=100,
        help="Number of epochs to train per BO trial"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of Bayesian optimization trials"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./checkpoints_bo",
        help="Directory to save BO results"
    )
    args = parser.parse_args()
    main(args)