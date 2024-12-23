import logging
import numpy as np
import torch
from tqdm import tqdm
from monai.metrics import compute_iou, compute_dice

from loss.losses import DICELoss, clipseg_loss
import torch.nn as nn


def compute_iou_(
    logits: torch.Tensor,
    gt_masks: torch.Tensor,
    texts: str,
    threshold: float = 0.1,
    eps: float = 1 - 6,
):
    # logits = torch.where(torch.sigmoid(logits) > 0.1, 1, 0)
    logits = torch.sigmoid(logits).squeeze(1)
    gt_masks = gt_masks.squeeze(1)
    intersection = (logits * gt_masks).float().sum((1, 2))
    union = (logits + gt_masks).float().sum((1, 2))
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def evaluate(adapted_clipseg, val_dataloader, epoch, num_epochs, device):
    adapted_clipseg.to(device)
    adapted_clipseg.eval()
    dice_loss = DICELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    losses, ious = [], []
    with torch.no_grad():
        logging.info(f"Launch Evaluation")
        with tqdm(
            total=len(val_dataloader), desc=f"Epoch {epoch}/{num_epochs}"
        ) as pbar:
            for _, (imgs, gt_masks, texts, inputs_ids, attn_masks) in enumerate(
                val_dataloader
            ):
                imgs = imgs.to(device)
                gt_masks = gt_masks.to(device)
                inputs_ids = inputs_ids.to(device).squeeze(1)
                attn_masks = attn_masks.to(device).squeeze(1)
                logits, visual_projection, textual_projection = adapted_clipseg(
                    texts=texts,
                    input_ids=inputs_ids,
                    pixel_values=imgs,
                    attention_mask=attn_masks,
                    gt_masks=gt_masks,
                )
                if len(imgs) == 1:
                    logits = logits.permute(1, 0, 2).unsqueeze(0)
                loss = (
                    bce_loss(logits, gt_masks.float())
                    + clipseg_loss(visual_projection, textual_projection)
                ) / 2.0
                # iou = compute_iou_(logits, gt_masks, texts)
                iou = compute_iou(logits, gt_masks, ignore_empty=False)
                losses.append(loss.item())
                ious.append(iou.mean().item())
                pbar.set_postfix({"Loss": loss.item(), "IoU": iou.mean().item()})
                pbar.update(1)
    return np.mean(losses), np.mean(ious)
