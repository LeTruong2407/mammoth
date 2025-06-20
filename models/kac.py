import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List

class KACClassifier(nn.Module):
    """
    Kolmogorov-Arnold Classifier (KAC) with RBF activations.
    Each input feature is passed through a learnable RBF for each class, then summed with learnable weights.
    Compatible with IncrementalClassifier interface.
    """
    def __init__(self, in_features, num_classes, num_rbfs=3, feat_expand=False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_rbfs = num_rbfs
        self.feat_expand = feat_expand

        # Create a ModuleList to hold KAC heads (one for each task)
        self.heads = nn.ModuleList()
        
        # Add the first head
        self.update(num_classes, freeze_old=False)
        
        self.old_state_dict = None

    def rbf(self, x, centers, widths):
        # x: (batch, in_features)
        # centers, widths: (num_classes, in_features, num_rbfs)
        # returns: (batch, num_classes, in_features, num_rbfs)
        x = x.unsqueeze(1).unsqueeze(-1)  # (batch, 1, in_features, 1)
        centers = centers.unsqueeze(0)    # (1, num_classes, in_features, num_rbfs)
        widths = widths.unsqueeze(0)      # (1, num_classes, in_features, num_rbfs)
        return torch.exp(-((x - centers) ** 2) / (2 * widths ** 2 + 1e-8))

    def enable_training(self):
        for p in self.parameters():
            p.requires_grad = True

    def disable_training(self):
        for p in self.parameters():
            p.requires_grad = False

    def build_optimizer_args(self, lr: float, wd: float = 0):
        params = []
        for ti in range(len(self.heads)):
            current_params = [p for p in self.heads[ti].parameters() if p.requires_grad]
            if len(current_params) > 0:
                params.append({
                    'params': current_params,
                    'lr': lr,
                    'weight_decay': wd
                })
        return params

    def assign(self, other):
        """
        Copy parameters from another KACClassifier.
        Handle cases where structures might be different.
        """
        # If the other classifier has a different structure, we need to be careful
        if isinstance(other, KACClassifier):
            # Copy the state dict, which should handle the parameter copying
            self.load_state_dict(other.state_dict())
        else:
            # Fallback: try to copy parameters directly
            for p_self, p_other in zip(self.parameters(), other.parameters()):
                if p_self.shape == p_other.shape:
                    p_self.data.copy_(p_other.data)

    def backup(self):
        """
        Save a backup of the current state dict.
        """
        if self.old_state_dict is not None:
            del self.old_state_dict
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        """
        Restore the state dict from the backup.
        """
        if hasattr(self, '_old_state_dict'):
            self.load_state_dict(self._old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        """
        Add a new KAC head for a new task.
        """
        # Create a new KAC layer
        new_head = KACLayer(self.in_features, nb_classes, self.num_rbfs)
        
        if freeze_old:
            self.disable_training()
        
        self.heads.append(new_head)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        return torch.cat(out, dim=1)

    def to_device(self, device):
        """
        Move all parameters to the specified device.
        """
        self.to(device)
        for head in self.heads:
            head.to(device)
        return self


class KACLayer(nn.Module):
    """
    Single KAC layer for one task.
    """
    def __init__(self, in_features, num_classes, num_rbfs=3):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_rbfs = num_rbfs

        # For each class and input feature, we have num_rbfs RBFs
        self.centers = nn.Parameter(torch.randn(num_classes, in_features, num_rbfs))
        self.widths = nn.Parameter(torch.ones(num_classes, in_features, num_rbfs))
        self.rbf_weights = nn.Parameter(torch.randn(num_classes, in_features, num_rbfs))
        self.out_bias = nn.Parameter(torch.zeros(num_classes))

    @property
    def weight(self):
        """
        Return weight in 2D format compatible with Fisher computation.
        Use the original rbf_weights shape but take the mean over num_rbfs dimension.
        """
        # Take mean over the num_rbfs dimension to get 2D tensor
        return self.rbf_weights.mean(dim=-1)  # [num_classes, in_features]

    @property
    def bias(self):
        """
        Return bias for Fisher computation compatibility.
        """
        return self.out_bias

    def rbf(self, x, centers, widths):
        # x: (batch, in_features)
        # centers, widths: (num_classes, in_features, num_rbfs)
        # returns: (batch, num_classes, in_features, num_rbfs)
        x = x.unsqueeze(1).unsqueeze(-1)  # (batch, 1, in_features, 1)
        centers = centers.unsqueeze(0)    # (1, num_classes, in_features, num_rbfs)
        widths = widths.unsqueeze(0)      # (1, num_classes, in_features, num_rbfs)
        return torch.exp(-((x - centers) ** 2) / (2 * widths ** 2 + 1e-8))

    def forward(self, x):
        """
        Forward pass with KAC computation.
        x: [batch_size, in_features]
        """
        # Get the 2D weight for computation
        weight_2d = self.weight  # [num_classes, in_features]
        
        # Compute linear transformation with 2D weight
        output = F.linear(x, weight_2d, self.bias)
        
        return output 
