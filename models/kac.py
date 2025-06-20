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
            # Ensure both are on the same device
            device = next(self.parameters()).device
            other = other.to(device)
            # Copy the state dict, which should handle the parameter copying
            self.load_state_dict(other.state_dict())
        else:
            # Fallback: try to copy parameters directly
            device = next(self.parameters()).device
            other = other.to(device)
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
        
        # Move new head to the same device as existing heads
        if len(self.heads) > 0:
            device = next(self.heads[0].parameters()).device
            new_head.to(device)
        
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
    def __init__(self, in_features, num_classes, num_rbfs=5):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_rbfs = num_rbfs

        # Better initialization for RBF parameters
        # Centers: initialize with small random values
        self.centers = nn.Parameter(torch.randn(num_classes, in_features, num_rbfs) * 0.1)
        # Widths: initialize with positive values
        self.widths = nn.Parameter(torch.ones(num_classes, in_features, num_rbfs) * 0.5)
        # RBF weights: initialize with small random values
        self.rbf_weights = nn.Parameter(torch.randn(num_classes, in_features, num_rbfs) * 0.1)
        # Output bias: initialize with zeros
        self.out_bias = nn.Parameter(torch.zeros(num_classes))

    @property
    def weight(self):
        """
        Return effective weight matrix for Fisher computation.
        Compute from RBF parameters to maintain compatibility.
        """
        # Compute effective weight by taking weighted sum of RBF centers
        # This approximates the linear transformation that RBFs perform
        effective_weight = (self.centers * self.rbf_weights).sum(dim=-1)  # [num_classes, in_features]
        return effective_weight

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
        Simplified KAC forward pass with memory efficiency.
        Uses linear transformation with RBF-inspired activation.
        x: [batch_size, in_features]
        """
        batch_size = x.size(0)
        
        # Use linear transformation as base
        # Compute effective weight matrix
        effective_weight = (self.centers * self.rbf_weights).sum(dim=-1)  # [num_classes, in_features]
        
        # Apply linear transformation
        logits = F.linear(x, effective_weight, self.out_bias)
        
        # Add RBF-inspired non-linearity (simplified)
        # This adds some non-linearity without the full RBF computation
        x_normalized = F.normalize(x, p=2, dim=1)
        weight_normalized = F.normalize(effective_weight, p=2, dim=1)
        
        # Add cosine similarity as additional feature
        cosine_sim = F.linear(x_normalized, weight_normalized, None)
        
        # Combine linear and cosine similarity
        logits = logits + 0.1 * cosine_sim
        
        return logits 
