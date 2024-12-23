import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np


class BatchVisualizer:
    def __init__(
        self,
        log_dir="/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/other/Unsupervised-Anomlay-Detection/roi/logs",
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def visualize_features(self, image_batch, n_iter, image_name="Image_batch"):
        grid = torchvision.utils.make_grid(image_batch)
        self.writer.add_image(image_name, grid, n_iter)

    def plot_loss(self, loss_val, n_iter, loss_name="loss"):
        self.writer.add_scalar(loss_name, loss_val, n_iter)
