import torch
import torch.nn as nn


class DICELoss(nn.Module):
    def __init__(self) -> None:
        super(DICELoss, self).__init__()

    def forward(self, logits, gt_masks):
        self.batch_size = logits.shape[0]
        logits = torch.sigmoid(logits).view(self.batch_size, -1)
        gt_masks = gt_masks.view(self.batch_size, -1)
        coeffs = 2 * (logits * gt_masks).sum(dim=1) / (logits + gt_masks).sum(dim=1)
        return 1 - coeffs.mean()


# contrastive loss function, adapted from https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(embedds: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        embedds, torch.arange(len(embedds), device=embedds.device)
    )


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->clipseg
def clipseg_loss(
    visual_projection: torch.Tensor, textual_projection: torch.Tensor
) -> torch.Tensor:
    visual_loss = contrastive_loss(visual_projection)
    textual_loss = contrastive_loss(textual_projection)
    return (visual_loss + textual_loss) / 2.0
