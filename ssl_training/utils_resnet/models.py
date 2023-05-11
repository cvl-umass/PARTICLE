from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn
import torchvision

class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


class Encoder(nn.Sequential):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = False) -> None:
        model = getattr(torchvision.models, backbone)(pretrained)
        self.emb_dim = model.fc.in_features
        model.fc = nn.Identity()
        model.avgpool = nn.Identity()
        super().__init__(*list(model.children()))


class MaskPooling(nn.Module):
    def __init__(
        self, num_classes: int, num_samples: int = 16, downsample: int = 32
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.mask_ids = torch.arange(num_classes)
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)

    def pool_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Create binary masks and performs mask pooling

        Args:
            masks: (b, 1, h, w)

        Returns:
            masks: (b, num_classes, d)
        """
        if masks.ndim < 4:
            masks = masks.unsqueeze(dim=1)

        masks = masks == self.mask_ids[None, :, None, None].to(masks.device)
        masks = self.pool(masks.to(torch.float))
        masks = masks.reshape(masks.shape[0], masks.shape[1],-1)
        # rearrange(masks, "b c h w -> b c (h w)")
        masks = torch.argmax(masks, dim=1)
        masks = torch.eye(self.num_classes).to(masks.device)[masks]
        masks = masks.permute(0,2,1)
        # rearrange(masks, "b d c -> b c d")
        return masks

    def sample_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Samples which binary masks to use in the loss.

        Args:
            masks: (b, num_classes, d)

        Returns:
            masks: (b, num_samples, d)
        """
        bs = masks.shape[0]
        mask_exists = torch.greater(masks.sum(dim=-1), 1e-3)
        sel_masks = mask_exists.to(torch.float) + 1e-11

        mask_ids = torch.multinomial(sel_masks, num_samples=self.num_samples)
        sampled_masks = torch.stack([masks[b][mask_ids[b]] for b in range(bs)])
        return sampled_masks, mask_ids

    def forward(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        binary_masks = self.pool_masks(masks)
        sampled_masks, sampled_mask_ids = self.sample_masks(binary_masks)
        area = sampled_masks.sum(dim=-1, keepdim=True)
        sampled_masks = sampled_masks / torch.maximum(area, torch.tensor(1.0))
        return sampled_masks, sampled_mask_ids


class Network(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = False,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_classes: int = 23,
        downsample: int = 32,
        num_samples: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(backbone, pretrained)
        state_dict = torch.load('./pretrained_models/resnet50_detcon.pth')['state_dict']
        self.encoder.load_state_dict(state_dict,strict=True)
        self.projector = MLP(self.encoder.emb_dim, hidden_dim, output_dim)
        self.mask_pool = MaskPooling(num_classes, num_samples, downsample)

    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> Sequence[torch.Tensor]:
        m, mids = self.mask_pool(masks)
        e = self.encoder(x)
        e = e.reshape(e.shape[0], e.shape[1],-1)
        e = e.permute(0,2,1)
        e = m @ e  
        p = self.projector(e)
        return e, p, m, mids
    
    def forward_discovery(self, x: torch.Tensor):
        e = self.encoder(x)
        e = e.reshape(e.shape[0], e.shape[1],-1)
        e = e.permute(0,2,1)
        p = self.projector(e)
        return p