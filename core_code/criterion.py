import torch

def stem_attachment_loss(
    coords: torch.Tensor,
    masks: torch.Tensor,
    num_masks: float,
    current_epoch: int,
    max_epochs: int,
    lambda_sa: float=0.05
):
    if isinstance(num_masks, torch.Tensor):
        num_masks_int = int(num_masks.item())
    else:
        num_masks_int = int(num_masks)

    z_coords = coords[:, 2].float()
    loss = []

    for i in range(num_masks_int):
        instance_mask = masks[i].bool()
        if instance_mask.any():
            instance_z = z_coords[instance_mask]
            k = max(1, int(len(instance_z) * 0.1))
            smallest_z = torch.topk(instance_z, min(k, len(instance_z)), largest=False)
            loss.append(smallest_z[0].sum())

    if len(loss) == 0:
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

    t = current_epoch
    T = max_epochs
    w_sa = 10.0 * t / T - 5.0
    gated_weight = lambda_sa * torch.sigmoid(torch.tensor(w_sa, device=coords.device))

    return gated_weight * (torch.stack(loss).sum() / num_masks_int)
