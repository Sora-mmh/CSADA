from typing import List
import math
from transformers import CLIPSegForImageSegmentation
from transformers.models.clipseg.modeling_clipseg import CLIPSegImageSegmentationOutput
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterModule(nn.Module):
    def __init__(
        self, input_dim: int, base_dim: int, hidden_vision_fuse: bool = False
    ) -> None:
        super().__init__()
        self.hidden_vision_fuse = hidden_vision_fuse
        self.adapt1 = nn.Linear(in_features=input_dim, out_features=base_dim * 4)
        self.adapt2 = nn.Linear(in_features=base_dim * 4, out_features=base_dim)
        self.adapt3 = nn.Linear(in_features=base_dim, out_features=base_dim * 8)
        self.adapt4 = nn.Linear(in_features=base_dim * 8, out_features=input_dim)
        self.activate = nn.ReLU()
        self.fuse = FusionModule(input_dim)

    def forward(
        self, x: torch.Tensor, hidden_states: List[torch.Tensor]
    ) -> torch.Tensor:
        out = self.activate(self.adapt1(x))
        out = self.activate(self.adapt2(out))
        out = self.activate(self.adapt3(out))
        out = self.activate(self.adapt4(out))
        if self.hidden_vision_fuse:
            return out + x + self.fuse(hidden_states)
        return out + x


class FusionModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        states_num: int = 13,
    ) -> None:
        super().__init__()
        self.base_dim = hidden_dim
        self.project = nn.Linear(
            in_features=states_num * self.base_dim, out_features=self.base_dim
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        hidden_states = torch.cat(hidden_states, dim=-1)
        return self.project(hidden_states)


class AdaptedCLIPSeg(nn.Module):
    def __init__(self, adapter_dim: int = 64) -> None:
        super().__init__()
        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.config = self.clipseg.config
        self.adapter_dim = adapter_dim
        self.vision_hidden_dim = self.config.vision_config.hidden_size
        self.language_hidden_dim = self.config.text_config.hidden_size
        self.conditional_embeddings_dim = self.config.projection_dim
        self.clipseg.requires_grad_(False)
        self.vision_adapters = nn.ModuleList(
            [
                AdapterModule(
                    input_dim=self.vision_hidden_dim,
                    base_dim=self.adapter_dim,
                    hidden_vision_fuse=True,
                )
                for _ in self.config.extract_layers
            ]
        )
        self.language_adapters = nn.ModuleList(
            [
                AdapterModule(
                    input_dim=self.language_hidden_dim, base_dim=self.adapter_dim
                )
                for _ in self.config.extract_layers
            ]
        )
        self.conditional_adapater = AdapterModule(
            input_dim=self.conditional_embeddings_dim, base_dim=self.adapter_dim
        )
        self.mask_patcher = nn.Conv2d(
            in_channels=1, out_channels=485, kernel_size=16, stride=16, padding=0
        )

        self.conditional_projection = nn.Linear(in_features=512, out_features=768)

    def extract_conditional_embeddings(
        self, texts, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        # with torch.no_grad():
        outputs = self.clipseg.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states, pooler_output = outputs[-1], outputs[1]
        self.conditional_embeddings = self.clipseg.clip.text_projection(pooler_output)
        self.conditional_embeddings = self.conditional_adapater(
            self.conditional_embeddings, hidden_states
        )
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
            hidden_state = language_adapter(hidden_state, hidden_states)
            self.conditional_embeddings += hidden_state

    def extract_vision_activations(
        self, texts, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        self.batch_size, self.chann_dim, self.height_dim, self.width_dim = (
            pixel_values.shape
        )
        # with torch.no_grad():
        outputs = self.clipseg.clip.vision_model(
            pixel_values=pixel_values, output_hidden_states=True
        )
        hidden_states = outputs[-1]  # outputs[-1] "hidden_states"
        self.vision_activations = [
            self.vision_adapters[idx](hidden_states[layer_idx + 1], hidden_states)
            for idx, layer_idx in enumerate(self.config.extract_layers)
        ]

        # self.conditional_embeddings_projection = self.conditional_projection(
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
        normalized_vision_activation = F.normalize(self.vision_activations[-1], dim=-1)
        normalized_vision_activation = normalized_vision_activation.transpose(-2, -1)
        text_activation = self.conditional_projection(self.conditional_embeddings)
        normalized_conditional_embeddings = F.normalize(text_activation, dim=-1)
        normalized_conditional_embeddings = normalized_conditional_embeddings.unsqueeze(
            1
        )
        activations = normalized_conditional_embeddings @ normalized_vision_activation
        activations = activations.squeeze()
        activations = F.sigmoid(activations * 10)

        decoded_outputs = self.clipseg.decoder(
            self.vision_activations,
            self.conditional_embeddings,
        )
        return (
            torch.sigmoid(decoded_outputs.logits.unsqueeze(1)),
            F.normalize(
                torch.sum(
                    self.vision_activations[-1] * activations.unsqueeze(-1), dim=1
                )
            ),
            F.normalize(text_activation),
        )

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


if __name__ == "__main__":
    adapted_clipseg = AdaptedCLIPSeg()
    pixel_values = torch.randn(1, 3, 256, 256)
    input_ids = torch.rand(1, 1).int()
    attention_mask = torch.randint(0, 2, (1, 1))
    gt_mask = torch.randn(1, 1, 256, 256)
    outputs = adapted_clipseg(input_ids, pixel_values, attention_mask, gt_mask)
    print("done")
