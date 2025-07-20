"""Training utilities for the chess network."""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    from onnxruntime.training import ORTModule
except Exception:  # pragma: no cover - optional dependency
    ORTModule = None
import wandb
from tqdm.auto import tqdm

from scripts.play_vs_ai import evaluate_against_previous

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


def _unwrap(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    if hasattr(model, "_original_module"):
        return model._original_module
    return model


def save_checkpoint(epoch, model, optimizer, scaler):
    path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch}.pt")
    base_model = _unwrap(model)
    torch.save(
        {
            "epoch": epoch,
            "model_state": base_model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        path,
    )
    print(f"⏺️ Checkpoint gespeichert: {path}")


def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location=Config.DEVICE)
    base_model = _unwrap(model)
    base_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["opt_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    print(f"⏺️ Checkpoint geladen: {path}")
    return ckpt["epoch"] + 1


from .config import Config


class Trainer:
    def __init__(
        self,
        network,
        buffer,
        optimizer,
        batch_size: int = Config.BATCH_SIZE,
        epochs: int = 1,
        log_dir: str = Config.LOG_DIR,
        use_wandb: bool = False,
    ):
        # 1) Auf GPU schieben …
        self.network = network.to(Config.DEVICE)
        # 2) TorchDynamo ohne Triton, mit aot_eager-Fallback:
        self.network = torch.compile(self.network, backend="aot_eager")
        if ORTModule is not None:
            self.network = ORTModule(self.network)
        # Gradient checkpointing handled inside the network's forward
        self.buffer = buffer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.writer = SummaryWriter(log_dir)
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=Config.WANDB_PROJECT)
            wandb.watch(self.network)

    def train(self):
        """Train the network using data from the replay buffer."""
        
        if len(self.buffer) < self.batch_size:
            return

        scaler = GradScaler(device="cuda")
        accumulation_steps = 4

        states, policies, values = self.buffer.sample(self.batch_size)
        states_np = np.stack(states, axis=0)
        states = torch.from_numpy(states_np).float()
        policies = torch.tensor(policies, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dataset = TensorDataset(states, policies, values)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        scheduler = CosineAnnealingLR(self.optimizer, T_max=len(loader) * self.epochs)

        start_epoch = 0
        if hasattr(Config, "RESUME_FROM") and Config.RESUME_FROM:
            start_epoch = load_checkpoint(
                Config.RESUME_FROM, self.network, self.optimizer, scaler
            )

        for epoch in range(start_epoch, self.epochs):
            epoch_loss = 0.0
            prog_bar = tqdm(
                loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch"
            )
            self.optimizer.zero_grad()
            for batch_idx, (s, p_target, v_target) in enumerate(prog_bar):
                s = s.cuda(non_blocking=True)
                p_target = p_target.cuda(non_blocking=True)
                v_target = v_target.cuda(non_blocking=True)
                with autocast(device_type="cuda"):
                    log_p, v = self.network(s)
                    loss_policy = -(p_target * log_p).sum(dim=1).mean()
                    loss_value = torch.mean((v.view(-1) - v_target) ** 2)
                    loss = (loss_policy + loss_value) / accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    scheduler.step()

                epoch_loss += loss.item() * accumulation_steps
                global_step = epoch * len(loader) + batch_idx
                self.writer.add_scalar(
                    "Loss/train", loss.item() * accumulation_steps, global_step
                )
                prog_bar.set_postfix(loss=loss.item() * accumulation_steps)

            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1}/{self.epochs} - loss {avg_loss:.4f}")
            self.writer.add_scalar("loss", avg_loss, epoch)
            if self.use_wandb:
                wandb.log({"loss": avg_loss})

            save_checkpoint(epoch, self.network, self.optimizer, scaler)

            win_rate, elo_delta = evaluate_against_previous(
                "C:\\Users\\Administrator\\Desktop\\Python\\SCHACHENGINE\\Test\\checkpoints\\latest.pt", games=100, simulations=25
            )
            print(
                f"\U0001f4ca Evaluation Epoche {epoch}: Win-Rate={win_rate:.2%}, ΔElo={elo_delta:.1f}"
            )
