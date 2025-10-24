import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

import const
from core import SynapseCore, SynapseMapping, Run, StepMetrics, SynapseConfig, TrainingStep
from gui_tensor import TensorModality
from models.retention import MultiScaleRetention

class SynapseNetAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture parameters
        hidden_dim = 256
        heads = 8
        ffn_size = 512
        enc_layers = 4
        dec_layers = 4

        # Encoder: transforms image into sequence
        self.to_seq = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # Encoder retention layers
        self.enc_retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim=True)
            for _ in range(enc_layers)
        ])
        self.enc_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(enc_layers)
        ])
        self.enc_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(enc_layers)])
        self.enc_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(enc_layers)])

        # Decoder retention layers
        self.dec_retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim=True)
            for _ in range(dec_layers)
        ])
        self.dec_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(dec_layers)
        ])
        self.dec_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(dec_layers)])
        self.dec_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(dec_layers)])

        # Decoder: transforms sequence back into image
        self.to_img = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

        # Optional classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 10)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.to_seq(x)
        x = x.flatten(2).transpose(1, 2)
        for i in range(len(self.enc_retentions)):
            y = self.enc_retentions[i](self.enc_norms1[i](x)) + x
            x = self.enc_ffns[i](self.enc_norms2[i](y)) + y
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.dec_retentions)):
            y = self.dec_retentions[i](self.dec_norms1[i](x)) + x
            x = self.dec_ffns[i](self.dec_norms2[i](y)) + y
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.to_img(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        logits = self.classifier(z.mean(dim=1))
        return recon, logits


@dataclass
class SynapseNetConfig(SynapseConfig):
    def create(self, *args, **kwargs) -> nn.Module:
        return SynapseNetAE()


class SynapsePetri(SynapseCore):
    def __init__(self, run: Run):
        super().__init__(run)

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = MNIST('./data', train=True, download=True, transform=self.transform)
        self.val_dataset = MNIST('./data', train=False, transform=self.transform)
        self.train_loader = self.create_training_loader(1)
        self.val_loader = self.create_validation_loader(1)

        self.recon_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

        self.mode = 'parallel'
        self.hidden_states = None

    def init_states(self, batch_size: int):
        """Initialize recurrent states for SynapseNet"""
        tissue = self.tissue
        device = next(tissue.parameters()).device
        enc_states, dec_states = [], []

        for retention in tissue.enc_retentions:
            layer_states = [torch.zeros(
                batch_size,
                retention.hidden_size // retention.heads,
                retention.v_dim // retention.heads,
                device=device
            ) for _ in range(retention.heads)]
            enc_states.append(layer_states)

        for retention in tissue.dec_retentions:
            layer_states = [torch.zeros(
                batch_size,
                retention.hidden_size // retention.heads,
                retention.v_dim // retention.heads,
                device=device
            ) for _ in range(retention.heads)]
            dec_states.append(layer_states)

        return enc_states, dec_states

    def train_step(self, step_num: int, step: TrainingStep) -> StepMetrics:
        imgs, labels = next(iter(self.train_loader))
        imgs, labels = imgs.to(const.device), labels.to(const.device)
        self.physics.zero_grad()

        if self.mode == 'parallel':
            recon, logits = self.tissue(imgs)
        else:
            if self.hidden_states is None:
                self.hidden_states = self.init_states(imgs.shape[0])
            enc_states, dec_states = self.hidden_states
            z = self.tissue.encode(imgs, enc_states)
            recon, logits = self.tissue.decode(z, dec_states)
            self.hidden_states = (enc_states, dec_states)

        recon_loss = self.recon_loss(recon, imgs)
        class_loss = self.class_loss(logits, labels)
        loss = recon_loss + 0.1 * class_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.tissue.parameters(), 1.0)
        self.physics.step()

        self.view('input', imgs[0].detach().cpu().numpy(), TensorModality.CHW_IMAGE_GRAYSCALE)
        self.view('output', recon[0].detach().cpu().numpy(), TensorModality.CHW_IMAGE_GRAYSCALE)
        mode_text = f"Mode: {self.mode}" + (f" (step {step_num})" if self.mode == 'recurrent' else "")
        self.view('mode_info', mode_text, TensorModality.TEXT)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()

        metrics = StepMetrics(
            loss=loss.item(),
            grad_norm=grad_norm.item(),
            param_norm=sum(p.norm().item() ** 2 for p in self.tissue.parameters()) ** 0.5,
            learning_rate=self.physics.param_groups[0]['lr'],
            batch_time=0.0,
            memory_used=torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0,
            accuracy=acc.item()
        )
        return metrics

    def validate(self) -> float:
        self.tissue.eval()
        total_loss, total_samples = 0, 0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(const.device), labels.to(const.device)
                recon, logits = self.tissue(imgs)
                loss = self.recon_loss(recon, imgs) + 0.1 * self.class_loss(logits, labels)
                total_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)

        return total_loss / total_samples

    def inference(self):
        """Run inference and visualize outputs."""
        self.tissue.eval()
        with torch.no_grad():
            imgs, labels = next(iter(self.val_loader))
            imgs = imgs.to(const.device)
            recon, logits = self.tissue(imgs)
            self.current_input = imgs[0].detach().cpu().numpy()
            self.output = recon[0].detach().cpu().numpy()


__synapse__ = SynapseMapping(
    tissue_class=SynapseNetAE,
    config_class=SynapseNetConfig,
    core_class=SynapsePetri,
)
