import logging
import random
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import lr_scheduler

# from adapter_v2 import AdapatedCLIPSeg
# from adapter import AdapatedCLIPSeg
from adapters.adapter_v3 import AdaptedCLIPSeg
from data.dataset import RS19Dataset
from loss.losses import DICELoss, clipseg_loss
from eval.metrics import evaluate
from utils.utils import BatchVisualizer

logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
debug = True


if __name__ == "__main__":
    adapted_clipseg = AdaptedCLIPSeg()
    train_dataset = RS19Dataset(mode="train")
    val_dataset = RS19Dataset(mode="val")
    test_dataset = RS19Dataset(mode="test")
    if debug:
        sampled_train_indices = random.sample(range(len(train_dataset)), 50)
        train_data = Subset(train_dataset, sampled_train_indices)
        sampled_val_indices = random.sample(range(len(val_dataset)), 10)
        val_data = Subset(val_dataset, sampled_val_indices)
        sampled_test_indices = random.sample(range(len(test_dataset)), 10)
        test_data = Subset(test_dataset, sampled_test_indices)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=16, shuffle=True, num_workers=8
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=16, shuffle=True, num_workers=8
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=16, shuffle=False, num_workers=8
    )
    num_epochs = 50

    optimizer = torch.optim.SGD(
        params=list(adapted_clipseg.vision_adapters.parameters())
        + list(adapted_clipseg.language_adapters.parameters())
        + list(adapted_clipseg.conditional_adapater.parameters()),
        lr=1e-4,
        momentum=0.9,
    )
    # optimizer = torch.optim.AdamW(
    #     params=list(adapted_clipseg.vision_adapters.parameters())
    #     + list(adapted_clipseg.language_adapters.parameters())
    #     + list(adapted_clipseg.conditional_adapater.parameters()),
    #     lr=1e-3,
    #     weight_decay=1e-4,
    # )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (1 - epoch / num_epochs) ** 0.5
    )
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DICELoss()
    adapted_clipseg.to(device)
    train_losses, val_losses, ious = [], [], []
    visualizer = BatchVisualizer()
    logging.info(f"Launch training for {num_epochs} epochs")
    for epoch in tqdm(range(1, num_epochs + 1)):
        adapted_clipseg.train()
        train_loss = 0.0
        with tqdm(
            total=len(train_dataloader), desc=f"Epoch {epoch}/{num_epochs}"
        ) as pbar:
            for i, (imgs, gt_masks, texts, input_ids, attn_masks) in enumerate(
                train_dataloader
            ):

                imgs = imgs.to(device)
                gt_masks = gt_masks.to(device)
                input_ids = input_ids.to(device).squeeze(1)
                attn_masks = attn_masks.to(device).squeeze(1)
                logits, visual_projection, textual_projection = adapted_clipseg(
                    texts=texts,
                    input_ids=input_ids,
                    pixel_values=imgs,
                    attention_mask=attn_masks,
                    gt_masks=gt_masks,
                )
                loss = (
                    bce_loss(logits, gt_masks.float())
                    + clipseg_loss(visual_projection, textual_projection)
                ) / 2.0

                # loss = (
                #     bce_loss(logits, gt_masks.float())
                #     + dice_loss(logits, gt_masks.float())
                # ) / 2
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                # if i == 80:
                # visualizer.plot_loss(loss, i, loss_name="Dice Loss")
                # if i == 80:
                # visualizer.visualize_features(gt_masks, i, image_name="Ground Truth")
                # visualizer.visualize_features(logits, i, image_name="Prediction")
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)
        scheduler.step()
        train_loss_per_epoch = train_loss / len(train_dataloader)
        # TODO: preprocess logits for evaluation(binary masks)
        val_loss_per_epoch, iou_per_epoch = evaluate(
            adapted_clipseg=adapted_clipseg,
            val_dataloader=val_dataloader,
            epoch=epoch,
            num_epochs=num_epochs,
            device=device,
        )
        train_losses.append(train_loss_per_epoch)
        val_losses.append(val_loss_per_epoch)
        ious.append(iou_per_epoch)
        plt.style.use("ggplot")
        figure = plt.gcf()
        figure.set_size_inches(5, 5)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].plot(train_losses)
        ax[0].set_title("Train Dice Loss")
        ax[1].plot(val_losses)
        ax[1].set_title("Eval Dice Loss")
        ax[2].plot(ious)
        ax[2].set_title("Eval IoU")
        monitor_training_pth = "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/other/Unsupervised-Anomlay-Detection/roi/monitoring/training.png"
        plt.savefig(monitor_training_pth, dpi=100)
        logging.info(f"Saving losses and metrics in {monitor_training_pth}")
        logging.info(
            f" Epoch {epoch}/{num_epochs}: Train Loss = {train_loss_per_epoch:.4f}   Eval Loss = {val_loss_per_epoch:.4f}   Eval IoU = {iou_per_epoch:.2f}"
        )
