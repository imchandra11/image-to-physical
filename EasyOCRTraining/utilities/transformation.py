import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TPS_SpatialTransformerNetwork(nn.Module):
    """ TPS-based Spatial Transformer for image rectification """
    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        # batch_I: [batch, C, H, W]
        batch_C_prime = self.LocalizationNetwork(batch_I)            # [batch, F, 2]
        batch_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # [batch, H_r*W_r, 2]
        H_r, W_r = self.I_r_size
        batch_P_prime = batch_P_prime.reshape(-1, H_r, W_r, 2)
        batch_I_r = F.grid_sample(batch_I, batch_P_prime, padding_mode='border')
        return batch_I_r

class LocalizationNetwork(nn.Module):
    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(self.I_channel_num, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # [batch, 512, 1, 1]
        )
        self.fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.fc2 = nn.Linear(256, 2 * F)
        # Initialize fc2 weights to zero and biases to the fiducial coordinates
        self.fc2.weight.data.fill_(0)
        import numpy as np
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F/2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F/2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F/2))
        initial_bias = np.concatenate([
            np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1),
            np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ], axis=0).reshape(-1)
        self.fc2.bias.data = torch.from_numpy(initial_bias).float()

    def forward(self, batch_I):
        features = self.conv(batch_I).view(batch_I.size(0), -1)
        fc1 = self.fc1(features)
        batch_C_prime = self.fc2(fc1).view(-1, self.F, 2)
        return batch_C_prime

class GridGenerator(nn.Module):
    def __init__(self, F, I_r_size):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        # Build fixed matrices for TPS
        C = self._build_C(F)   # [F,2]
        P = self._build_P(self.I_r_width, self.I_r_height)  # [n,2]
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(F, C), dtype=torch.float))  # [F+3, F+3]
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(F, C, P), dtype=torch.float))            # [n, F+3]

    def _build_C(self, F):
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F/2))
        ctrl_pts_y_top = -1 * np.ones(int(F/2))
        ctrl_pts_y_bottom = np.ones(int(F/2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # [F,2]

    def _build_inv_delta_C(self, F, C):
        # Compute the inverse of the TPS matrix delta_C
        hat_C = np.zeros((F, F), dtype=np.float32)
        for i in range(F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i,j] = r
                hat_C[j,i] = r
        np.fill_diagonal(hat_C, 1)  # avoid log(0)
        hat_C = (hat_C**2) * np.log(hat_C)
        delta_C = np.concatenate([
            np.concatenate([np.ones((F,1)), C, hat_C], axis=1),      # [F, F+3]
            np.concatenate([np.zeros((2,3)), C.T], axis=1),         # [2, F+3]
            np.concatenate([np.zeros((1,3)), np.ones((1,F))], axis=1)  # [1, F+3]
        ], axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # [F+3, F+3]

    def _build_P(self, I_r_width, I_r_height):
        # Generate grid points P (flattened)
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        P = np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)  # [H, W, 2]
        P = P.reshape(-1, 2)
        return P  # [n, 2]

    def _build_P_hat(self, F, C, P):
        # Build matrix P_hat for transformation
        n = P.shape[0]
        # pairwise distances for RBF
        P_tile = np.tile(P[:, None, :], (1, F, 1))  # [n,F,2]
        C_tile = np.expand_dims(C, 0)              # [1,F,2]
        P_diff = P_tile - C_tile  # [n, F, 2]
        rbf_norm = np.linalg.norm(P_diff, axis=2)  # [n, F]
        rbf = (rbf_norm**2) * np.log(rbf_norm + self.eps)
        P_hat = np.concatenate([np.ones((n,1)), P, rbf], axis=1)  # [n, F+3]
        return P_hat

    def build_P_prime(self, batch_C_prime):
        # Given predicted fiducials C' (batch x F x 2), compute grid transform
        batch_size = batch_C_prime.size(0)
        inv_delta_C = self.inv_delta_C.unsqueeze(0).repeat(batch_size,1,1)  # [batch, F+3, F+3]
        P_hat = self.P_hat.unsqueeze(0).repeat(batch_size,1,1)              # [batch, n, F+3]
        # Append zeros to C' to match size (F+3)x2
        zeros = torch.zeros(batch_size, 3, 2, device=batch_C_prime.device)
        C_prime_with_zeros = torch.cat([batch_C_prime, zeros], dim=1)  # [batch, F+3, 2]
        # Compute transformation matrix T = inv_delta_C * C'
        batch_T = torch.bmm(inv_delta_C, C_prime_with_zeros)  # [batch, F+3, 2]
        # Compute P' = P_hat * T
        batch_P_prime = torch.bmm(P_hat, batch_T)  # [batch, n, 2]
        return batch_P_prime
