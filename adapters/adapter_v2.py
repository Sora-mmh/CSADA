from typing import List
from transformers import CLIPSegForImageSegmentation
from transformers.models.clipseg.modeling_clipseg import CLIPSegImageSegmentationOutput
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterModule(nn.Module):
    def __init__(self, hidden_dim):
        super(AdapterModule, self).__init__()
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim**-0.5

    def forward(self, x: torch.Tensor):
        q, k, v = self.Q(x), self.K(x), self.V(x)
        if len(x.shape) == 2:
            attn_scores = (
                torch.bmm(q.unsqueeze(1), k.unsqueeze(1).transpose(1, 2)) * self.scale
            )
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.bmm(attn_weights, v.unsqueeze(1))
            return attn_output.squeeze(1), attn_weights.squeeze(1)
        else:
            attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.bmm(attn_weights, v)
            return attn_output, attn_weights


class AdapatedCLIPSeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.config = self.clipseg.config
        self.vision_hidden_dim = self.config.vision_config.hidden_size
        self.language_hidden_dim = self.config.text_config.hidden_size
        self.conditional_embeddings_dim = self.config.projection_dim
        self.clipseg.requires_grad_(False)
        self.vision_adapters = nn.ModuleList(
            [
                AdapterModule(hidden_dim=self.vision_hidden_dim)
                for _ in self.config.extract_layers
            ]
        )
        self.language_adapters = nn.ModuleList(
            [
                AdapterModule(hidden_dim=self.language_hidden_dim)
                for _ in self.config.extract_layers
            ]
        )
        self.conditional_adapater = AdapterModule(
            hidden_dim=self.conditional_embeddings_dim
        )
        # self.conditional_projection = nn.Linear(in_features=512, out_features=768)

    def extract_conditional_embeddings(
        self, texts, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        outputs = self.clipseg.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states, pooler_output = outputs[-1], outputs[1]
        self.conditional_embeddings = self.clipseg.clip.text_projection(pooler_output)
        # self.conditional_embeddings, _ = self.conditional_adapater(
        #     self.conditional_embeddings
        # )
        for layer_idx, language_adapter in zip(
            self.config.extract_layers, self.language_adapters
        ):
            hidden_state = hidden_states[layer_idx + 1]
            hidden_state = hidden_state[
                torch.arange(hidden_state.shape[0], device=hidden_state.device),
                input_ids.to(dtype=torch.int, device=hidden_state.device).argmax(
                    dim=-1
                ),
            ]
            # hidden_state, _ = language_adapter(hidden_state)
            self.conditional_embeddings += hidden_state

    def extract_vision_activations(
        self, texts, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        self.batch_size, self.chann_dim, self.height_dim, self.width_dim = (
            pixel_values.shape
        )
        outputs = self.clipseg.clip.vision_model(
            pixel_values=pixel_values, output_hidden_states=True
        )
        hidden_states = outputs[-1]
        self.vision_activations = [
            self.vision_adapters[idx](hidden_states[layer_idx + 1])[0]
            for idx, layer_idx in enumerate(self.config.extract_layers)
        ]
        # projected_conditional_embeddings = self.conditional_projection(
        #     self.conditional_embeddings
        # ).unsqueeze(1)
        # for idx in range(len(self.config.extract_layers)):
        #     hidden_state = self.vision_activations.pop(0)
        #     scores = torch.bmm(
        #         hidden_state, projected_conditional_embeddings.transpose(1, 2)
        #     ).squeeze(-1)
        #     attn = nn.functional.softmax(scores, dim=1)
        #     hidden_state *= attn.unsqueeze(-1)
        #     self.vision_activations.append(
        #         self.vision_adapters[idx](hidden_state, all_hidden_states)
        #     )

    def forward(
        self,
        texts,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:
        self.extract_conditional_embeddings(
            texts=texts, input_ids=input_ids, attention_mask=attention_mask
        )
        self.extract_vision_activations(texts=texts, pixel_values=pixel_values)
        decoded_outputs = self.clipseg.decoder(
            self.vision_activations,
            self.conditional_embeddings,
        )
        # logits = decoded_outputs.logits.view(
        #     self.batch_size, 1, self.height_dim, self.width_dim
        # )
        return decoded_outputs.logits.unsqueeze(1)
        # if gt_masks is not None:
        #     criterion = nn.BCEWithLogitsLoss()
        #     loss = criterion(logits, gt_masks.float())
        #     return loss, logits

        # return CLIPSegImageSegmentationOutput(
        #     loss=loss,
        #     logits=logits,
        #     conditional_embeddings=self.conditional_embeddings,
        #     pooled_output=pooled_output,
        #     vision_model_output=self.vision_activations,
        #     decoder_output=decoded_outputs,
        # )


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->clipseg
def clipseg_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


if __name__ == "__main__":
    adapted_clipseg = AdapatedCLIPSeg()
    pixel_values = torch.randn(1, 3, 256, 256)
    input_ids = torch.rand(1, 1).int()
    attention_mask = torch.randint(0, 2, (1, 1))
    gt_mask = torch.randn(1, 1, 256, 256)
    outputs = adapted_clipseg(input_ids, pixel_values, attention_mask, gt_mask)

    # va = AdapterModule(hidden_dim=768)
    # rd = torch.randn(16, 485, 768)
    # out = va(rd)
    print("done")
