from typing import Callable
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from transformers import CLIPTokenizer, AutoTokenizer


class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        except:
            logging.info(
                "Invalid_transpose, please make sure images have shape (H, W, C) before transposing"
            )
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.600, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


def get_data_transforms(size=(352, 352)):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.600, 0.225]
    img_transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size),
            T.Normalize(mean=mean, std=std),
        ]
    )
    mask_transform = T.Compose([T.ToTensor(), T.Resize(size)])
    return img_transform, mask_transform


def extract_cls_from_json(config_json: dict):
    return {cls["name"]: idx for idx, cls in enumerate(config_json["labels"])}


# transforms: Callable[[any], any], phase: str


# {'road': 0,
#  'sidewalk': 1,
#  'construction': 2,
#  'tram-track': 3,
#  'fence': 4,
#  'pole': 5,
#  'traffic-light': 6,
#  'traffic-sign': 7,
#  'vegetation': 8,
#  'terrain': 9,
#  'sky': 10,
#  'human': 11,
#  'rail-track': 12,
#  'car': 13,
#  'truck': 14,
#  'trackbed': 15,
#  'on-rails': 16,
#  'rail-raised': 17,
#  'rail-embedded': 18}


# 0, 5, 8, 10, 12,    |3, 4, 6, 7, 11, 13, 14, 15
class RS19Dataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        modes = ["train", "val", "test"]
        assert self.mode in modes, logging.info(f"Loading mode must be in {[modes]}")
        self.mode_split_pth = Path(
            "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/data/wilddash2/rs19_splits4000"
        ) / (mode + ".txt")
        with open(self.mode_split_pth, "r") as f:
            stems = f.read().splitlines()
        self.dir_pth = Path(
            "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/data/wilddash2"
        )
        self.name = "rs19_val"
        self.dataset_pth = self.dir_pth / self.name
        self.imgs_pths = [
            pth
            for pth in list((self.dataset_pth / "jpgs" / self.name).iterdir())
            if pth.stem in stems
        ]
        self.imgs_pths = sorted(self.imgs_pths)
        self.masks_pths = [
            pth
            for pth in list((self.dataset_pth / "uint8" / self.name).iterdir())
            if pth.stem in stems
        ]
        self.masks_pths = sorted(self.masks_pths)
        self.dataset_config = json.load(
            open(self.dir_pth / self.name / "rs19-config.json", "r")
        )
        self.cls = extract_cls_from_json(self.dataset_config)
        self.img_transform, self.mask_transform = get_data_transforms()
        self.tokenizer = CLIPTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.prepare_language_vision_inputs()

    def prepare_language_vision_inputs(self):
        self.imgs, self.masks, self.texts, self.inputs_ids, self.attn_masks = (
            [],
            [],
            [],
            [],
            [],
        )
        for img_pth, mask_pth in tqdm(
            zip(self.imgs_pths[:100], self.masks_pths[:100]),
            desc=f"{self.mode.upper()} Dataset Generation",
        ):
            gt_masks = cv2.imread(mask_pth.as_posix(), 0)
            for text, idx in tqdm(self.cls.items(), desc=f"Image {img_pth.name}"):
                mask = gt_masks == idx
                if mask.any() and idx in [5, 8, 10, 12]:
                    tokenized_text = self.tokenizer(
                        text=text,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt",
                    )
                    # input_ids = torch.tensor(
                    #     self.tokenizer.encode(text, add_special_tokens=True)
                    # )
                    # attention_mask = torch.ones((1, 3))
                    input_ids = tokenized_text["input_ids"]
                    attention_mask = tokenized_text["attention_mask"]
                    self.imgs.append(img_pth)
                    self.masks.append(mask)
                    self.texts.append(f"{text}")
                    self.inputs_ids.append(input_ids)
                    self.attn_masks.append(attention_mask)

    def __getitem__(self, idx: int):
        img_pth, mask, text, input_ids, attention_mask = (
            self.imgs[idx],
            self.masks[idx],
            self.texts[idx],
            self.inputs_ids[idx],
            self.attn_masks[idx],
        )
        mask = mask.astype(int)
        img = cv2.imread(img_pth.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (
            self.img_transform(img),
            self.mask_transform(mask),
            text,
            input_ids,
            attention_mask,
        )

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    data = RS19Dataset(mode="val")
    dataloader = DataLoader(data, batch_size=16, shuffle=True)
    for imgs, masks, inputs_ids, attn_masks in dataloader:
        print("debugging")
