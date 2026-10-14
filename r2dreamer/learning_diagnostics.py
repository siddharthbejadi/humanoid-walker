"""Opt-in representation loss and diagnostics for the walking experiment.

The feature-mean reduction divides the *whole* Barlow objective by D. Averaging
the diagonal and off-diagonal separately would also change their relative weight.
"""
import torch


def barlow_objective(projected, target, coefficient, reduction="sum", stride=1):
    if reduction not in ("sum", "feature_mean") or stride < 1:
        raise ValueError("Invalid Barlow reduction/stride")
    # Preserve temporal sequences elsewhere; subsample only this objective.
    x = projected[:, ::stride].reshape(-1, projected.shape[-1]).float()
    y = target[:, ::stride].reshape(-1, target.shape[-1]).detach().float()
    if x.shape != y.shape:
        raise ValueError("Barlow branches must have matching shapes")
    if x.shape[0] < 2:
        x = projected.reshape(-1, projected.shape[-1]).float()
        y = target.reshape(-1, target.shape[-1]).detach().float()
    if x.shape[0] < 2:
        raise ValueError("Barlow requires at least two batch-time samples")
    # Population variance agrees with the 1/N correlation denominator. Compute
    # in FP32 even when the rest of the model uses automatic mixed precision.
    with torch.autocast(device_type=x.device.type, enabled=False):
        sx, sy = x.std(0, correction=0), y.std(0, correction=0)
        nx = (x - x.mean(0)) / sx.clamp_min(1e-4)
        ny = (y - y.mean(0)) / sy.clamp_min(1e-4)
        c = nx.T @ ny / x.shape[0]
        diag = c.diagonal()
        inv = (diag - 1).square().sum()
        off = c.square().sum() - diag.square().sum()
        off = off.clamp_min(0)
        loss = inv + coefficient * off
        if reduction == "feature_mean":
            loss = loss / x.shape[-1]
        metrics = {
            "barlow/samples": loss.new_tensor(x.shape[0]),
            "barlow/dimensions": loss.new_tensor(x.shape[-1]),
            "barlow/diag_mean": diag.mean().detach(),
            "barlow/diag_error": (inv / x.shape[-1]).detach(),
            "barlow/offdiag_rms": (off / max(1, x.shape[-1] * (x.shape[-1] - 1))).sqrt().detach(),
            "barlow/x1_std": sx.mean().detach(),
            "barlow/x2_std": sy.mean().detach(),
            "barlow/x1_low_variance_fraction": (sx < 1e-4).float().mean().detach(),
            "barlow/x2_low_variance_fraction": (sy < 1e-4).float().mean().detach(),
        }
    return loss, metrics


@torch.no_grad()
def sampled_effective_rank(features):
    """Rank of a fixed bounded subsample, not the full embedding covariance."""
    x = features.detach().float().reshape(-1, features.shape[-1])
    rows = torch.linspace(0, x.shape[0] - 1, min(64, x.shape[0]), device=x.device).long()
    cols = torch.linspace(0, x.shape[1] - 1, min(128, x.shape[1]), device=x.device).long()
    x = x[rows][:, cols]
    singular = torch.linalg.svdvals(x - x.mean(0))
    total = singular.sum()
    p = singular / total.clamp_min(1e-12)
    rank = (-(p * p.clamp_min(1e-12).log()).sum()).exp()
    return torch.where(total > 1e-12, rank, torch.zeros_like(rank))
