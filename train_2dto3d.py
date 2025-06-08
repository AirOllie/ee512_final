#!/usr/bin/env python3

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------------------
# 1. H36MDataset
# ----------------------------------------------------------------------
class H36MDataset(Dataset):
    def __init__(self, data_path, split="train", image_size=(1000, 1000)):
        """
        data_path: root of H36M (must contain annot/train.h5, annot/valid.h5, train_images.txt, valid_images.txt)
        split: "train" or "valid"
        image_size: (H, W) for normalizing 2D → [-1,+1].
        """
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

        # Load image paths (we only need them for visualization)
        with open(img_list_file, "r") as f:
            self.image_names = [line.strip() for line in f]

        # Load raw 2D (part) and 3D (S) from HDF5
        with h5py.File(h5_file, "r") as f:
            raw_2d = f["part"][:]    # shape: (N, 2, 17) or (N,17,2)
            raw_3d = f["S"][:]       # shape: (N, 3, 17) or (N,17,3)

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
        """
        Returns:
          keypoints2d: FloatTensor(17,2) in [-1,+1]
          keypoints3d: FloatTensor(17,3), root‐centered at pelvis (joint 0).
        """
        k2d = self.joints_2d[idx].copy()  # (17,2) in pixels
        k3d = self.joints_3d[idx].copy()  # (17,3) in meters

        H, W = self.image_size

        # Normalize 2D: (x,y) pixel → (x_norm, y_norm) in [-1,+1]
        x = k2d[:, 0]
        y = k2d[:, 1]
        x_n = (x - (W / 2.0)) / (W / 2.0)
        y_n = - (y - (H / 2.0)) / (H / 2.0)  # flip so that "up" is +1
        keypoints2d = np.stack([x_n, y_n], axis=1).astype(np.float32)  # (17,2)

        # Root‐center 3D: subtract joint 0 (pelvis)
        root = k3d[0:1, :]               # (1,3)
        centered_3d = (k3d - root).astype(np.float32)  # (17,3)

        return torch.from_numpy(keypoints2d), torch.from_numpy(centered_3d)

# ----------------------------------------------------------------------
# 2. Skeleton connectivity & adjacency normalization (17 joints).
# ----------------------------------------------------------------------
# H36M joint indices:
# 0: Pelvis, 1: Right Hip, 2: Right Knee, 3: Right Foot
# 4: Left Hip, 5: Left Knee, 6: Left Foot
# 7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head
# 11: Left Shoulder, 12: Left Elbow, 13: Left Wrist
# 14: Right Shoulder, 15: Right Elbow, 16: Right Wrist

H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),        # Right leg
    (0, 4), (4, 5), (5, 6),        # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine → head
    (11, 12), (12, 13),            # Left arm
    (14, 15), (15, 16),            # Right arm
    (11, 8), (8, 14)               # Shoulders → thorax
]

def build_adjacency(num_joints=17, skeleton=H36M_SKELETON):
    """
    Builds a symmetric adjacency matrix A (17×17) with 1s on edges + self‐loops.
    """
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for (u, v) in skeleton:
        A[u, v] = 1.0
        A[v, u] = 1.0
    for i in range(num_joints):
        A[i, i] = 1.0
    return A

def normalize_adjacency(A):
    """
    Symmetrically normalize A: A_norm = D^{-0.5} A D^{-0.5}.
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=(D>0))
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat
    return torch.from_numpy(A_norm.astype(np.float32))

# ----------------------------------------------------------------------
# 3. GCN model
# ----------------------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, A_norm):
        """
        x: (B, num_nodes, in_channels)
        A_norm: (num_nodes, num_nodes)
        Returns: (B, num_nodes, out_channels)
        """
        # Aggregation: A_norm @ x  → (B, num_nodes, in_channels)
        x_agg = torch.einsum("ij,bjk->bik", A_norm, x)
        out = self.fc(x_agg)
        return out

class Pose2MeshJoints(nn.Module):
    def __init__(self, A_norm, hidden_dims=[64, 128, 128, 64, 32]):
        """
        A_norm: torch.Tensor (17×17) normalized adjacency
        hidden_dims: [64, 128, 128, 64, 32]
        """
        super(Pose2MeshJoints, self).__init__()
        self.A_norm = A_norm  # (17×17)

        # Input MLP: 2 → 64
        self.lin_in = nn.Linear(2, hidden_dims[0])

        # GraphConv blocks
        self.gcns = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gcns.append(GraphConv(hidden_dims[i], hidden_dims[i+1]))

        # BatchNorm for each hidden block (except the first projection & final output)
        self.bns = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_dims[1:]])

        # Final GraphConv: 32 → 3
        self.gc_out = GraphConv(hidden_dims[-1], 3)

        self.relu = nn.ReLU()

    def forward(self, x2d):
        """
        x2d: (B, 17, 2)
        Returns: x3d: (B, 17, 3)
        """
        B, N, _ = x2d.shape  # N==17

        # (1) Project 2D → 64
        h = self.lin_in(x2d)  # (B,17,64)
        h = self.relu(h)

        # (2) GCN blocks with BN + ReLU
        for idx, gcn in enumerate(self.gcns):
            h = gcn(h, self.A_norm)  # (B,17, hidden_dims[idx+1])
            bn = self.bns[idx]       # BatchNorm1d(hidden_dims[idx+1])
            # BN expects (B, C, N), so transpose
            h = bn(h.transpose(1, 2)).transpose(1, 2)
            h = self.relu(h)

        # (3) Final 3D prediction
        x3d = self.gc_out(h, self.A_norm)  # (B,17,3)
        return x3d

# ----------------------------------------------------------------------
# 4. Geometric constraint losses
# ----------------------------------------------------------------------
def compute_angle_loss(p1, p2, p3):
    """
    Compute the angle at p2 formed by p1-p2-p3.
    Returns cos(angle) which is 1 for straight line (180 degrees).
    """
    v1 = p1 - p2
    v2 = p3 - p2
    v1_norm = F.normalize(v1, dim=-1)
    v2_norm = F.normalize(v2, dim=-1)
    cos_angle = torch.sum(v1_norm * v2_norm, dim=-1)
    return cos_angle

def compute_plane_normal(p1, p2, p3):
    """
    Compute normal vector of plane defined by three points.
    """
    v1 = p2 - p1
    v2 = p3 - p1
    normal = torch.cross(v1, v2, dim=-1)
    return F.normalize(normal, dim=-1)

def compute_bone_length_ratios(pred3d):
    """
    Compute bone length ratios based on reference measurements.
    Reference lengths (in cm):
    - Shoulder → Elbow (upper arm): 36.90 cm
    - Elbow → Wrist (forearm): 29.03 cm
    - Hip → Knee (thigh): 61.64 cm
    - Knee → Ankle (shank): 40.50 cm
    - Left shoulder → Right shoulder: 45.0 cm
    - Left hip → Right hip: 35.3 cm
    - Shoulder → Hip (torso): 38.37 cm
    """
    # Joint indices
    pelvis, r_hip, r_knee, r_foot = 0, 1, 2, 3
    l_hip, l_knee, l_foot = 4, 5, 6
    spine, thorax, neck, head = 7, 8, 9, 10
    l_shoulder, l_elbow, l_wrist = 11, 12, 13
    r_shoulder, r_elbow, r_wrist = 14, 15, 16
    
    # Reference ratios (normalized by upper arm length as base = 1.0)
    base_length = 0.369  # 36.90 cm
    ref_ratios = {
        'upper_arm': 1.0,  # 36.90 / 36.90
        'forearm': 0.787,  # 29.03 / 36.90
        'thigh': 1.671,  # 61.64 / 36.90
        'shank': 1.098,  # 40.50 / 36.90
        'shoulder_width': 1.220,  # 45.0 / 36.90
        'hip_width': 0.957,  # 35.3 / 36.90
        'torso': 1.040,  # 38.37 / 36.90
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
    
    # For torso, use average of shoulder positions to average of hip positions
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

def compute_geometric_constraints(pred3d, epoch):
    """
    pred3d: (B,17,3)
    epoch: int
    Returns losses dict and weights dict.
    """
    B = pred3d.shape[0]
    losses, weights = {}, {}

    # only start at epoch 51
    if epoch <= 50:
        return losses, weights
    ramp = min((epoch - 50) / 50.0, 1.0)

    # indices
    pel, r_hip, r_knee, r_foot = 0, 1, 2, 3
    l_hip, l_knee, l_foot = 4, 5, 6
    spine, thorax, neck, head = 7, 8, 9, 10
    ls, le, lw = 11, 12, 13
    rs, re, rw = 14, 15, 16

    # reusable angle‐cos
    def cos_ang(a,b,c):
        return compute_angle_loss(a,b,c)  # gives cos(angle)

    # --- Strong constraints (high weight) ---
    # 1) hip chain straight: angle > 90° ⇒ cos(angle)<0 ⇒ penalize if cos>0
    hip_cos = cos_ang(pred3d[:, l_hip], pred3d[:, pel], pred3d[:, r_hip])
    loss = torch.mean(F.relu(hip_cos))
    losses['strong_hip_straight'] = loss
    weights['strong_hip_straight'] = 0.3 * ramp

    # 2) shoulder chain >90°
    sh_cos = cos_ang(pred3d[:, ls], pred3d[:, thorax], pred3d[:, rs])
    loss = torch.mean(F.relu(sh_cos))
    losses['strong_shoulder_angle'] = loss
    weights['strong_shoulder_angle'] = 0.25 * ramp

    # 3) neck‐thorax‐spine >90°
    nts_cos = cos_ang(pred3d[:, neck], pred3d[:, thorax], pred3d[:, spine])
    loss = torch.mean(F.relu(nts_cos))
    losses['strong_neck_thorax_spine'] = loss
    weights['strong_neck_thorax_spine'] = 0.25 * ramp

    # 4) head‐neck‐thorax >90°
    hnt_cos = cos_ang(pred3d[:, head], pred3d[:, neck], pred3d[:, thorax])
    loss = torch.mean(F.relu(hnt_cos))
    losses['strong_head_neck_thorax'] = loss
    weights['strong_head_neck_thorax'] = 0.2 * ramp

    # 5) never curve backward in thorax-spine-pelvis
    front_n = compute_coronal_plane_normal(pred3d)
    # vector at spine = bisector of (thorax→spine) & (pelvis→spine)
    v1 = pred3d[:, thorax] - pred3d[:, spine]
    v2 = pred3d[:, pel]    - pred3d[:, spine]
    bis = F.normalize(v1 + v2, dim=-1)
    # dot(bis, front_n) should be ≥ 0 ⇒ penalize bis·front_n < 0
    loss = torch.mean(F.relu(-torch.sum(bis * front_n, dim=-1)))
    losses['strong_thorax_spine_pelvis'] = loss
    weights['strong_thorax_spine_pelvis'] = 0.2 * ramp

    # 6) hip‐knee‐foot never curve in front (separating line dot front_n ≤ 0)
    v1 = pred3d[:, l_hip] - pred3d[:, l_knee]
    v2 = pred3d[:, l_foot]- pred3d[:, l_knee]
    bis_l = F.normalize(v1 + v2, dim=-1)
    loss_l = torch.mean(F.relu(torch.sum(bis_l * front_n, dim=-1)))
    v1 = pred3d[:, r_hip] - pred3d[:, r_knee]
    v2 = pred3d[:, r_foot]- pred3d[:, r_knee]
    bis_r = F.normalize(v1 + v2, dim=-1)
    loss_r = torch.mean(F.relu(torch.sum(bis_r * front_n, dim=-1)))
    losses['strong_hip_knee_foot'] = 0.5 * (loss_l + loss_r)
    weights['strong_hip_knee_foot'] = 0.2 * ramp


    # --- Medium constraints (mid weight) ---
    # shoulder straight (weakly)
    loss = -torch.mean(cos_ang(pred3d[:, ls], pred3d[:, thorax], pred3d[:, rs]))
    losses['med_shoulder_straight'] = loss
    weights['med_shoulder_straight'] = 0.15 * ramp

    # neck‐thorax‐spine straight
    loss = -torch.mean(cos_ang(pred3d[:, neck], pred3d[:, thorax], pred3d[:, spine]))
    losses['med_neck_spine_straight'] = loss
    weights['med_neck_spine_straight'] = 0.15 * ramp

    # bone‐length ratios
    bone_loss = compute_bone_length_ratios(pred3d)
    losses['med_bone_ratios'] = bone_loss
    weights['med_bone_ratios'] = 0.15 * ramp


    # --- Weak constraints (low weight) ---
    # vertical spine‐hip‐pelvis / pelvis‐hip‐knee
    vert = torch.tensor([0, -1, 0], device=pred3d.device).float()
    # pelvis→hip
    ph = pred3d[:, l_hip] - pred3d[:, pel]
    sp = pred3d[:, spine] - pred3d[:, pel]
    w1 = 1 - torch.abs(F.cosine_similarity(ph, vert.unsqueeze(0), dim=-1))
    w2 = 1 - torch.abs(F.cosine_similarity(sp, vert.unsqueeze(0), dim=-1))
    losses['weak_spine_vertical'] = torch.mean(w1 + w2)
    weights['weak_spine_vertical'] = 0.1 * ramp

    hk = pred3d[:, l_knee] - pred3d[:, l_hip]
    w3 = 1 - torch.abs(F.cosine_similarity(hk, vert.unsqueeze(0), dim=-1))
    hk2= pred3d[:, r_knee] - pred3d[:, r_hip]
    w4 = 1 - torch.abs(F.cosine_similarity(hk2, vert.unsqueeze(0), dim=-1))
    losses['weak_hip_knee_vertical'] = torch.mean(w3 + w4)
    weights['weak_hip_knee_vertical'] = 0.1 * ramp

    # coronal‐orientation: wrists, knees, head should lie on “positive” side
    # wrists
    ws_l = pred3d[:, lw] - ((pred3d[:, ls] + pred3d[:, rs]) / 2)
    ws_r = pred3d[:, rw] - ((pred3d[:, ls] + pred3d[:, rs]) / 2)
    loss = torch.mean(F.relu(-torch.sum(ws_l * front_n,    dim=-1)))
    loss += torch.mean(F.relu(-torch.sum(ws_r * front_n,    dim=-1)))
    # knees
    ks_l = pred3d[:, l_knee] - ((pred3d[:, l_hip] + pred3d[:, l_foot]) / 2)
    ks_r = pred3d[:, r_knee] - ((pred3d[:, r_hip] + pred3d[:, r_foot]) / 2)
    loss += torch.mean(F.relu(-torch.sum(ks_l * front_n,    dim=-1)))
    loss += torch.mean(F.relu(-torch.sum(ks_r * front_n,    dim=-1)))
    # head
    hs = pred3d[:, head] - ((pred3d[:, neck] + pred3d[:, thorax]) / 2)
    loss += torch.mean(F.relu(-torch.sum(hs  * front_n,    dim=-1)))
    losses['weak_coronal_orientation'] = loss / 5.0
    weights['weak_coronal_orientation'] = 0.1 * ramp

    return losses, weights

def compute_coronal_plane_normal(pred3d):
    """
    pred3d: (B,17,3)
    Coronal plane is defined by left_shoulder (11), right_shoulder (14), pelvis (0).
    Returns: (B,3) unit normal, pointing roughly “forward”.
    """
    ls = pred3d[:, 11]   # left shoulder
    rs = pred3d[:, 14]   # right shoulder
    pel = pred3d[:, 0]   # pelvis
    shoulder_center = (ls + rs) / 2.0

    # vector along shoulders:
    v1 = rs - ls
    # vector from shoulder_center to pelvis:
    v2 = pel - shoulder_center

    # normal = v1 × v2
    n = torch.cross(v1, v2, dim=-1)
    return F.normalize(n, dim=-1)

# ----------------------------------------------------------------------
# 5. Plotting / visualization utilities
# ----------------------------------------------------------------------
def build_image_lookup(root_dir):
    """
    Build a dict mapping each image basename → absolute path (walks subfolders).
    """
    lookup = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                lookup[fname] = os.path.join(dirpath, fname)
    return lookup

def plot_2d_on_axis(ax, image, keypoints, skeleton, radius=1, color=(0,1,0), linewidth=2):
    """
    Draw 2D joints & skeleton on ax (imshow).
    """
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
    """
    Rotate 3D points by -90° about X-axis: (x,y,z) → (x,z,-y).
    joints_3d: np.ndarray (17,3)
    """
    x = joints_3d[:,0]; y = joints_3d[:,1]; z = joints_3d[:,2]
    return np.stack([ x, z, -y ], axis=1)

def rotate_combined(joints_3d):
    """
    Only -90° about X
    """
    return rotate_x_minus90(joints_3d)

def plot_3d_on_axis(ax3d, joints_3d, skeleton, title=None):
    """
    Draw 3D skeleton (blue) on a 3D axis. 
    Converts to mm if in meters, centers at pelvis, and applies rotation.
    """
    pts = joints_3d.copy()
    if np.max(np.abs(pts)) < 10:  # meters → mm
        pts = pts * 1000.0
    # Center at pelvis (joint 0)
    centered = pts - pts[0]
    rotated = rotate_combined(centered)

    # Scatter & bones in blue
    ax3d.scatter(rotated[:,0], rotated[:,1], rotated[:,2], c="b", s=30)
    for (p, c) in skeleton:
        ax3d.plot(
            [rotated[p,0], rotated[c,0]],
            [rotated[p,1], rotated[c,1]],
            [rotated[p,2], rotated[c,2]],
            c="b", linewidth=2
        )
    ax3d.set_xlabel("X (mm)"); ax3d.set_ylabel("Y (mm)"); ax3d.set_zlabel("Z (mm)")
    if title:
        ax3d.set_title(title)
    xyz_min = np.min(rotated, axis=0)
    xyz_max = np.max(rotated, axis=0)
    max_range = np.max(xyz_max - xyz_min)/2.0
    mid_x = (xyz_max[0] + xyz_min[0])/2.0
    mid_y = (xyz_max[1] + xyz_min[1])/2.0
    mid_z = (xyz_max[2] + xyz_min[2])/2.0
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

# ----------------------------------------------------------------------
# 6. MPJPE metric & train/val loops with tqdm
# ----------------------------------------------------------------------
def mpjpe(pred, target):
    """
    Mean Per‐Joint Position Error (meters). pred,target: (B,17,3)
    """
    return torch.mean(torch.norm(pred - target, dim=2))

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_constraint_loss = 0.0
    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    for k2d, k3d in loop:
        k2d = k2d.to(device)  # (B,17,2)
        k3d = k3d.to(device)  # (B,17,3)

        optimizer.zero_grad()
        pred3d = model(k2d)   # (B,17,3)
        
        # Main reconstruction loss
        recon_loss = F.mse_loss(pred3d, k3d)
        
        # Geometric constraints (activated after epoch 50)
        constraint_losses, constraint_weights = compute_geometric_constraints(pred3d, epoch)
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
        loop.set_postfix(loss=recon_loss.item(), constraint=constraint_loss.item(), mpjpe=(batch_mpjpe*1000.0))

    N = len(dataloader.dataset)
    return total_loss / N, total_mpjpe / N, total_constraint_loss / N

def validate_one_epoch(model, dataloader, device, epoch):
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    loop = tqdm(dataloader, desc=f"Valid Epoch {epoch}", leave=False)
    with torch.no_grad():
        for k2d, k3d in loop:
            k2d = k2d.to(device)
            k3d = k3d.to(device)
            pred3d = model(k2d)

            loss = F.mse_loss(pred3d, k3d)
            batch_mpjpe = mpjpe(pred3d, k3d).item()

            total_loss += loss.item() * k2d.shape[0]
            total_mpjpe += batch_mpjpe * k2d.shape[0]
            loop.set_postfix(val_loss=loss.item(), val_mpjpe=(batch_mpjpe*1000.0))

    N = len(dataloader.dataset)
    return total_loss / N, total_mpjpe / N

# ----------------------------------------------------------------------
# 7. Visualization (every 5 epochs, run on a few train samples)
# ----------------------------------------------------------------------
def visualize_samples(model, dataset, image_lookup, device, epoch, vis_dir, num_samples=5):
    """
    For a few random training indices, run the model and save a figure showing:
      - Left: 2D keypoints on the RGB image
      - Right: GT 3D skeleton (blue) and predicted 3D skeleton (red)
    Saves to vis_dir/epoch_{epoch:03d}_idx_{idx}.png
    """
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        k2d_norm, k3d_gt = dataset[idx]  # k2d_norm: (17,2), k3d_gt: (17,3)
        with torch.no_grad():
            pred3d = model(k2d_norm.unsqueeze(0).to(device))  # (1,17,3)
        pred3d = pred3d.cpu().squeeze(0).numpy()  # (17,3)
        gt3d = k3d_gt.numpy()                     # (17,3)

        # Raw 2D pixel coords for plotting
        raw_k2d = dataset.joints_2d[idx]          # (17,2) pixels

        # Find the corresponding RGB image
        relpath = dataset.image_names[idx]
        basename = os.path.basename(relpath)
        img_path = image_lookup.get(basename)
        if img_path is None or not os.path.isfile(img_path):
            print(f"[Warning] Image not found for {basename}, skipping idx {idx}.")
            continue
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Plot side-by-side
        fig = plt.figure(figsize=(16, 8))

        # Left: 2D overlay
        ax2d = fig.add_subplot(1, 2, 1)
        plot_2d_on_axis(
            ax2d,
            img_rgb,
            raw_k2d,
            H36M_SKELETON,
            radius=1,
            color=(0, 1, 0),
            linewidth=2
        )
        ax2d.set_title(f"Epoch {epoch} — Train idx={idx} — 2D Keypoints")

        # Right: 3D GT (blue) & Pred (red)
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
        plot_3d_on_axis(ax3d, gt3d, H36M_SKELETON, title="GT (blue) & Pred (red) 3D Keypoints")

        # Overlay prediction in red
        pred = pred3d.copy()
        if np.max(np.abs(pred)) < 10:  # meters → mm
            pred = pred * 1000.0
        pred_centered = pred - pred[0]
        rot_pred = rotate_combined(pred_centered)

        ax3d.scatter(rot_pred[:,0], rot_pred[:,1], rot_pred[:,2], c="r", s=30, marker="o", label="Pred")
        for (p, c) in H36M_SKELETON:
            ax3d.plot(
                [rot_pred[p,0], rot_pred[c,0]],
                [rot_pred[p,1], rot_pred[c,1]],
                [rot_pred[p,2], rot_pred[c,2]],
                c="r", linewidth=2
            )
        ax3d.legend()

        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"epoch_{epoch:03d}_idx_{idx}.png")
        plt.savefig(save_path)
        plt.close(fig)

# ----------------------------------------------------------------------
# 8. Main: parse args, create datasets, model, optimizer, scheduler, train.
# ----------------------------------------------------------------------
def main(args):
    # Always use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Build & normalize adjacency exactly as in Pose2Mesh
    A = build_adjacency(num_joints=17)
    A_norm = normalize_adjacency(A).to(device)  # (17×17)

    # Create train/valid datasets & loaders
    train_dataset = H36MDataset(args.data_path, split="train", image_size=args.image_size)
    valid_dataset = H36MDataset(args.data_path, split="valid", image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Instantiate model
    model = Pose2MeshJoints(A_norm, hidden_dims=args.hidden_dims).to(device)

    # Optimizer: Adam(lr=1e-3, weight_decay=0)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    # LR scheduler: drop LR by 0.1 at epochs 80 and 140 (total 200 epochs)
    scheduler = MultiStepLR(optimizer, milestones=[80, 140], gamma=0.1)

    # Prepare image lookup for visualization
    images_dir = os.path.join(args.data_path, "images")
    image_lookup = build_image_lookup(images_dir)

    # Prepare output directories
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    best_val_mpjpe = float("inf")

    train_losses, val_losses = [], []
    train_mpjpes,  val_mpjpes  = [], []

    # === Training loop (200 epochs total) ===
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mpjpe, train_constraint = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_mpjpe = validate_one_epoch(model, valid_loader, device, epoch)

        print(
            f"Epoch {epoch:03d}/200 | "
            f"Train Loss: {train_loss:.4f} | Train MPJPE: {train_mpjpe*1000:.2f} mm | "
            f"Constraint: {train_constraint:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MPJPE: {val_mpjpe*1000:.2f} mm"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mpjpes.append(train_mpjpe * 1000.0)  # to mm
        val_mpjpes.append(val_mpjpe   * 1000.0)

        # Step the LR scheduler (exactly at the end of each epoch)
        scheduler.step()

        # Save best model (by val MPJPE)
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            best_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  → New best (Val MPJPE={val_mpjpe*1000:.2f} mm). Saved to {best_path}")

        # Every 5 epochs, do a quick visualization on some train samples:
        if epoch % 5 == 0:
            print(f"[Info] Saving visualization for epoch {epoch} ...")
            visualize_samples(model, train_dataset, image_lookup, device, epoch, vis_dir, num_samples=5)

    # Save final checkpoint
    final_path = os.path.join(args.out_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[Info] Training complete. Final model saved to {final_path}")

    epochs = list(range(1, args.epochs+1))
    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"))
    plt.close()

    # MPJPE curve
    plt.figure()
    plt.plot(epochs, train_mpjpes, label="Train MPJPE (mm)")
    plt.plot(epochs, val_mpjpes,   label="Val MPJPE (mm)")
    plt.xlabel("Epoch"); plt.ylabel("MPJPE (mm)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "mpjpe_curve.png"))
    plt.close()

    print(f"[Info] Plotted training curves to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str,
        default="/home/oliver/Documents/data/h36m",
        help="Root directory of Human3.6M (must contain annot/ and images/)"
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[1000, 1000],
        help="(H, W) used to normalize 2D keypoints"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial LR = 1e-3")
    parser.add_argument(
        "--hidden_dims", type=int, nargs="+",
        default=[64, 128, 128, 64, 32],
        help="Hidden dims for each GraphConv block"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Total epochs"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./checkpoints_customize_pro",
        help="Directory to save best/final model and visualizations"
    )
    args = parser.parse_args()
    main(args)