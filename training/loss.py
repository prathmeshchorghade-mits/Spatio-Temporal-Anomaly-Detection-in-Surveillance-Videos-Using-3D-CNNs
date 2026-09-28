import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class MILRankingLoss(nn.Module):
    """
    Multiple Instance Learning (MIL) ranking loss for video anomaly detection.
    Computes ranking loss, sparsity loss, and temporal smoothness loss.
    """
    def __init__(
        self,
        margin: float = 0.5,
        lambda_sparse: float = 8e-5,
        lambda_smooth: float = 8e-5,
        topk: Optional[int] = None
    ):
        super().__init__()
        self.margin = margin
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.topk = topk

    def forward(
        self,
        anomaly_logits: torch.Tensor,
        normal_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the MIL loss.
        
        Args:
            anomaly_logits: Tensor of shape (B, T) containing logits for anomalous video bags.
            normal_logits: Tensor of shape (B, T) containing logits for normal video bags.
            
        Returns:
            Dictionary containing total loss and individual components.
        """
        assert anomaly_logits.dim() == 2, f"Expected anomaly_logits to have 2 dims, got {anomaly_logits.dim()}"
        assert normal_logits.dim() == 2, f"Expected normal_logits to have 2 dims, got {normal_logits.dim()}"
        
        # Calculate scores for sparsity and smoothness (requires sigmoid)
        anomaly_scores = torch.sigmoid(anomaly_logits)
        
        # 1. Ranking Loss (uses logits to avoid saturation)
        if self.topk is None:
            # max-MIL
            anom_top_k, _ = torch.max(anomaly_logits, dim=1)
            norm_top_k, _ = torch.max(normal_logits, dim=1)
        else:
            # top-k MIL
            anom_top_k = torch.topk(anomaly_logits, self.topk, dim=1)[0].mean(dim=1)
            norm_top_k = torch.topk(normal_logits, self.topk, dim=1)[0].mean(dim=1)
            
        # L_rank = max(0, margin - topk_anom + topk_norm)
        rank_loss = F.relu(self.margin - anom_top_k + norm_top_k).mean()
        
        # 2. Sparsity Loss (only on anomalous bags)
        sparse_loss = self.lambda_sparse * anomaly_scores.mean()
        
        # 3. Smoothness Loss (only on anomalous bags)
        smooth_loss = self.lambda_smooth * torch.abs(
            anomaly_scores[:, 1:] - anomaly_scores[:, :-1]
        ).mean()
        
        total_loss = rank_loss + sparse_loss + smooth_loss
        
        return {
            'total': total_loss,
            'rank': rank_loss,
            'sparse': sparse_loss,
            'smooth': smooth_loss
        }

def _smoke_test():
    print("Running MILRankingLoss smoke test...")
    B, T = 4, 32
    
    # Dummy logits
    anomaly_logits = torch.randn(B, T, requires_grad=True)
    normal_logits = torch.randn(B, T, requires_grad=True)
    
    # Test with max-MIL
    print("\nTesting max-MIL (topk=None)...")
    loss_fn_max = MILRankingLoss(topk=None)
    out_max = loss_fn_max(anomaly_logits, normal_logits)
    
    for k, v in out_max.items():
        print(f"{k}: {v.item():.4f}")
        
    out_max['total'].backward()
    assert anomaly_logits.grad is not None and normal_logits.grad is not None
    print("Gradients computed successfully for max-MIL.")
    
    # Zero grads
    anomaly_logits.grad.zero_()
    normal_logits.grad.zero_()
    
    # Test with top-k MIL
    print("\nTesting top-k MIL (topk=3)...")
    loss_fn_topk = MILRankingLoss(topk=3)
    out_topk = loss_fn_topk(anomaly_logits, normal_logits)
    
    for k, v in out_topk.items():
        print(f"{k}: {v.item():.4f}")
        
    out_topk['total'].backward()
    assert anomaly_logits.grad is not None and normal_logits.grad is not None
    print("Gradients computed successfully for top-k MIL.")
    
    print("\nSmoke test passed!")

if __name__ == "__main__":
    _smoke_test()
