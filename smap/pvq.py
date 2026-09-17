"""Patch VQ-VAE.

Trains a single shared codebook over normalized local complex STFT patches
extracted around each peak (anchor + target). Patch = peak-centered window
`(2*rf+1) x (2*rt+1)`, normalized by dividing by the peak complex value so the
center pixel is 1+0j. Representation = real+imag stacked as two channels.

Model: CNN encoder → Euclidean VQ → CNN decoder. Reconstruction L1 on complex.
Decoder only used during training (to learn a useful latent); at inference the
codebook centroids (stored as normalized patches) are pasted directly onto the
STFT canvas, no neural forward pass.
"""


import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, IterableDataset
from smap.utils import set_torch_device
from smap.config import (
    PATCH_DICT_SIZE,
    PATCH_H,
    PATCH_W,
    PVQ_BATCH_SIZE,
    PVQ_EPOCHS,
    PVQ_LR,
    PVQ_MAX_BATCH_EPOCH,
    PVQ_MIN_LR,
    PVQ_N_STAGES,
)


class PatchVQDataset(IterableDataset):
    """Multi-chunk shuffle buffer iterable over HDF5 complex patches.

    Random per-sample reads ~1 TB/epoch (too slow). Single-chunk reads correlate
    batches (same audio file per chunk → biased gradients, loss oscillation).
    Fix: read K random chunks into buffer, shuffle union, yield. Decorrelated.
    Per epoch: buf_reads × K chunks × 2.13 MB. K=16 buffer = ~34 MB per refill."""
    def __init__(self, dataset_path: str, samples_per_epoch: int, shuffle_chunks: int = 16) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.samples_per_epoch = samples_per_epoch
        self.shuffle_chunks = shuffle_chunks
        with h5py.File(dataset_path, "r") as d:
            ds = d["patches"]
            self.length = ds.shape[0]
            self.chunk_size = ds.chunks[0] if ds.chunks else 4096
        self.n_chunks = (self.length + self.chunk_size - 1) // self.chunk_size
        print(f"[INFO] PVQ dataset: {self.length} patches, {self.n_chunks} chunks of {self.chunk_size}, buf={self.shuffle_chunks}")

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        f = h5py.File(self.dataset_path, "r", swmr=True)
        ds = f["patches"]
        yielded = 0
        try:
            while yielded < self.samples_per_epoch:
                blocks = []
                for _ in range(self.shuffle_chunks):
                    ci = int(np.random.randint(0, self.n_chunks))
                    s = ci * self.chunk_size
                    e = min(s + self.chunk_size, self.length)
                    blocks.append(ds[s:e])
                buf = np.concatenate(blocks, axis=0)
                order = np.random.permutation(buf.shape[0])
                for j in order:
                    if yielded >= self.samples_per_epoch:
                        break
                    yield torch.from_numpy(buf[j]).float()
                    yielded += 1
        finally:
            f.close()

def energy_weighted_l1(recon: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L1 weighted by |target| magnitude. Borders with low |z| carry random phase
    (different peak-to-peak, unlearnable). At inference, max-|z| paste means
    borders superpose and high-|z| pixels dominate. Train for what matters."""
    mag = torch.sqrt(target[:, 0:1] ** 2 + target[:, 1:2] ** 2)  # (B, 1, H, W)
    err = torch.abs(recon - target)  # (B, 2, H, W)
    num = (err * mag).sum()
    den = mag.sum() * err.shape[1] + eps
    return num / den


class VectorQuantPatch(nn.Module):
    """Euclidean VQ with straight-through estimator (same as scalar VQ in cbook)."""
    def __init__(self, dict_size: int, embedding_dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.codebook = nn.Embedding(dict_size, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / embedding_dim, 1.0 / embedding_dim)

    def forward(self, z):
        distances = torch.cdist(z, self.codebook.weight, p=2.0)
        token_ids = torch.argmin(distances, dim=1)
        zq = self.codebook.weight[token_ids]
        loss_cb = nn.functional.mse_loss(zq, z.detach())
        loss_commit = self.beta * nn.functional.mse_loss(z, zq.detach())
        loss_vq = loss_cb + loss_commit
        zq_st = z + (zq - z).detach()
        return zq_st, token_ids, loss_vq


class PatchVQFlat(nn.Module):
    """Residual Vector Quantization in patch space — no encoder, no decoder.

    Standard audio-codec RVQ (SoundStream/EnCodec). Each stage quantizes residual
    from previous stage. Effective capacity = N_STAGES * log2(dict_size) bits.
    Zero transformation error; accumulates quantization over N stages.

    Codebook updated by gradient on per-stage codebook MSE vs residual.
    """
    def __init__(self, dict_size: int, n_stages: int = PVQ_N_STAGES) -> None:
        super().__init__()
        self.dict_size = dict_size
        self.n_stages = n_stages
        self.patch_dim = 2 * PATCH_H * PATCH_W
        self.stages = nn.ModuleList([
            nn.Embedding(dict_size, self.patch_dim) for _ in range(n_stages)
        ])
        # Stage 0 init: center 1+0j (mean patch). Subsequent: small random.
        # dim_scale keeps code ‖·‖ ~constant across patch sizes.
        center_idx_real = PATCH_H // 2 * PATCH_W + PATCH_W // 2
        center_idx_imag = PATCH_H * PATCH_W + center_idx_real
        dim_scale = (130.0 / self.patch_dim) ** 0.5
        with torch.no_grad():
            nn.init.normal_(self.stages[0].weight, mean=0.0, std=0.1 * dim_scale)
            self.stages[0].weight[:, center_idx_real] = 1.0
            self.stages[0].weight[:, center_idx_imag] = 0.0
            for s in range(1, n_stages):
                nn.init.normal_(self.stages[s].weight, mean=0.0, std=(0.02 / (s + 1)) * dim_scale)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W) → tokens (B, n_stages)"""
        b = x.shape[0]
        xf = x.reshape(b, -1)
        residual = xf.clone()
        tokens = []
        for stage in self.stages:
            dist = torch.cdist(residual, stage.weight, p=2.0)
            tok = torch.argmin(dist, dim=1)
            code = stage.weight[tok]
            residual = residual - code
            tokens.append(tok)
        return torch.stack(tokens, dim=1)  # (B, n_stages)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, n_stages) → (B, 2, H, W)"""
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(-1)
        out = 0
        for i, stage in enumerate(self.stages):
            out = out + stage.weight[token_ids[:, i]]
        return out.reshape(-1, 2, PATCH_H, PATCH_W)

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        xf = x.reshape(b, -1)
        residual = xf.detach()
        recon_flat = torch.zeros_like(xf)
        tokens = []
        cb_loss = torch.zeros((), device=x.device)

        for stage in self.stages:
            with torch.no_grad():
                dist = torch.cdist(residual, stage.weight, p=2.0)
                tok = torch.argmin(dist, dim=1)
            code = stage.weight[tok]
            cb_loss = cb_loss + nn.functional.mse_loss(code, residual.detach())
            residual = (residual - code).detach()
            recon_flat = recon_flat + code.detach()
            tokens.append(tok)

        recon = recon_flat.reshape(-1, 2, PATCH_H, PATCH_W)
        recon_loss = nn.functional.l1_loss(recon, x)
        total = cb_loss
        token_ids_stacked = torch.stack(tokens, dim=1)
        return recon, token_ids_stacked, total, recon_loss, cb_loss


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride=(1, 1)) -> None:
        super().__init__()
        num_groups = min(8, c_out)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class _ConvTBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride=(1, 1)) -> None:
        super().__init__()
        num_groups = min(8, c_out)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, output_padding=0),
            nn.GroupNorm(num_groups, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        num_groups = min(8, c)
        self.net = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, c),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class PatchEnCodec(nn.Module):
    """CNN VQ-VAE on (2, PATCH_H, PATCH_W) tensors.

    Encoder: Conv + ResBlocks at each scale. Freq stride (2,1) twice: 13→7→4.
    Latent = flatten × Linear(latent_dim). Decoder mirrors with ConvTranspose.
    """
    def __init__(self, latent_dim: int, dict_size: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.h_mid = PATCH_H  # 13
        self.w_mid = PATCH_W  # 5
        # freq: 13 → 7 → 4
        self.enc = nn.Sequential(
            _ConvBlock(2, 64),
            _ResBlock(64),
            _ConvBlock(64, 128, stride=(2, 1)),
            _ResBlock(128),
            _ConvBlock(128, 256, stride=(2, 1)),
            _ResBlock(256),
        )
        self.enc_h = (PATCH_H + 1) // 2  # 7
        self.enc_h = (self.enc_h + 1) // 2  # 4
        self.enc_flat = 256 * self.enc_h * PATCH_W
        self.enc_fc = nn.Linear(self.enc_flat, latent_dim)

        self.vq = VectorQuantPatch(dict_size=dict_size, embedding_dim=latent_dim)

        self.dec_fc = nn.Linear(latent_dim, self.enc_flat)
        self.dec_c = 256
        self.dec = nn.Sequential(
            _ResBlock(256),
            _ConvTBlock(256, 128, stride=(2, 1)),
            _ResBlock(128),
            _ConvTBlock(128, 64, stride=(2, 1)),
            _ResBlock(64),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = h.reshape(h.shape[0], -1)
        return self.enc_fc(h)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z)
        h = h.reshape(-1, self.dec_c, self.enc_h, PATCH_W)
        return self.dec(h)

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        z = self.vq.codebook.weight[token_ids]
        return self.decode_latent(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        zq, token_ids, loss_vq = self.vq(z)
        recon = self.decode_latent(zq)
        recon = recon[:, :, :PATCH_H, :PATCH_W]
        recon_loss = nn.functional.l1_loss(recon, x)
        total = recon_loss + loss_vq
        return recon, token_ids, total, recon_loss, loss_vq


class PatchVQ:
    def __init__(self) -> None:
        dev = set_torch_device()
        self.device = torch.device(dev)
        self.dataloader = None
        self.model: PatchVQFlat | None = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.prev_avg_loss = float("inf")

    def load_dataset(self, dataset_path: str) -> None:
        nsamples = PVQ_MAX_BATCH_EPOCH * PVQ_BATCH_SIZE
        ds = PatchVQDataset(dataset_path=dataset_path, samples_per_epoch=nsamples)
        self.dataloader = DataLoader(dataset=ds, batch_size=PVQ_BATCH_SIZE, num_workers=0)

    def net_init(self) -> None:
        self.model = PatchVQFlat(dict_size=PATCH_DICT_SIZE).to(self.device)
        self.optimizer = Adam(params=self.model.parameters(), lr=PVQ_LR)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer, T_max=PVQ_EPOCHS, eta_min=PVQ_MIN_LR
        )

    @torch.no_grad()
    def kmeans_init(self, n_samples: int = 65536, n_iters: int = 5) -> None:
        """Proper k-means codebook init (Lloyd's algorithm) per RVQ stage.

        1. Collect pool of real patches
        2. Per stage: random-seed centroids, then Lloyd iterations (assign → mean update)
        3. Compute residual after converged stage, feed to next stage

        Random-sample init gave poor partitions (1024 random points don't cover
        65k-patch distribution). Lloyd iters move centroids to local cluster means
        → tight partitions → small residuals → monotonic exponential decay."""
        if self.model is None or self.dataloader is None:
            print("[ERROR] Null model or dataloader — cannot init.")
            return
        print(f"[INFO] kmeans_init: collecting {n_samples} patches...")
        collected = []
        count = 0
        for batch in self.dataloader:
            collected.append(batch.to(self.device).reshape(batch.shape[0], -1))
            count += batch.shape[0]
            if count >= n_samples:
                break
        pool = torch.cat(collected, dim=0)[:n_samples]
        n, d = pool.shape
        k = self.model.dict_size
        print(f"[INFO] kmeans_init: pool {pool.shape}, {n_iters} Lloyd iters × {self.model.n_stages} stages...")
        residual = pool.clone()
        ones = torch.ones(n, device=self.device)
        for s in range(self.model.n_stages):
            stage = self.model.stages[s]
            # Seed: k random points from current residual
            perm = torch.randperm(n, device=self.device)[:k]
            centroids = residual[perm].clone()
            # Lloyd iterations
            for it in range(n_iters):
                dist = torch.cdist(residual, centroids, p=2.0)
                tok = torch.argmin(dist, dim=1)
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(k, device=self.device)
                new_centroids.index_add_(0, tok, residual)
                counts.index_add_(0, tok, ones)
                mask = counts > 0
                new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
                # Empty clusters: resample from residual to preserve k coverage
                n_empty = int((~mask).sum().item())
                if n_empty > 0:
                    replace_idx = torch.randperm(n, device=self.device)[:n_empty]
                    new_centroids[~mask] = residual[replace_idx]
                centroids = new_centroids
            stage.weight.data.copy_(centroids)
            # Final assignment + residual update
            dist = torch.cdist(residual, centroids, p=2.0)
            tok = torch.argmin(dist, dim=1)
            residual = residual - centroids[tok]
            rnorm = residual.norm(dim=1).mean().item()
            n_used = int(torch.unique(tok).numel())
            print(f"  stage {s:2d}: residual ‖·‖ = {rnorm:.4f}  used {n_used}/{k}")
        print("[INFO] kmeans_init done.")

    def train(self) -> None:
        root = Path("./sonarmap").resolve().parent
        out_dir = root.joinpath("models").joinpath("vq")
        out_dir.mkdir(exist_ok=True, parents=True)

        if self.model is None:
            print("[ERROR] Null model. Call net_init() first!")
            return

        dlen = len(self.dataloader)
        try:
            for epoch in tqdm(range(PVQ_EPOCHS), desc="TRAIN PVQ"):
                self.model.train()
                epoch_total = 0.0
                epoch_recon = 0.0
                epoch_vq = 0.0

                used = torch.zeros(self.model.n_stages, PATCH_DICT_SIZE, dtype=torch.bool, device=self.device)
                active_patch_buffer = []

                real_epoch = self.start_epoch + epoch
                for i, batch in enumerate(self.dataloader):
                    batch = batch.to(self.device)

                    recon, token_ids, total, recon_loss, vq_loss = self.model(batch)
                    self.optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # track usage per stage
                    for s in range(self.model.n_stages):
                        used[s, token_ids[:, s]] = True
                    if len(active_patch_buffer) < 1024:
                        active_patch_buffer.append(batch.detach().reshape(batch.shape[0], -1))

                    epoch_total += total.item()
                    epoch_recon += recon_loss.item()
                    epoch_vq += vq_loss.item()

                    if i % 100 == 0:
                        print(f"Epoch [{real_epoch + 1}/{PVQ_EPOCHS}] | Batch [{i}/{dlen}] | "
                              f"Total {total.item():.4f} | Recon {recon_loss.item():.4f} | VQ {vq_loss.item():.4f}")

                dead_per_stage = (~used).sum(dim=1)
                dead_total = dead_per_stage.sum().item()
                avg_total = epoch_total / max(1, dlen)
                avg_recon = epoch_recon / max(1, dlen)
                avg_vq = epoch_vq / max(1, dlen)
                print(f"=== End Epoch {real_epoch + 1} | Loss {avg_total:.4f} | "
                      f"Recon {avg_recon:.4f} | VQ {avg_vq:.4f} | Dead {dead_total} ({dead_per_stage.tolist()}) ===")
                if self.scheduler is not None:
                    self.scheduler.step()

                # Revive dead codes: replace with random patches from active buffer.
                if dead_total > 0 and len(active_patch_buffer) > 0:
                    with torch.no_grad():
                        all_patches = torch.cat(active_patch_buffer, dim=0)
                        if all_patches.shape[0] > 4096:
                            idx = torch.randperm(all_patches.shape[0], device=self.device)[:4096]
                            all_patches = all_patches[idx]
                        residual = all_patches.clone()
                        for s in range(self.model.n_stages):
                            stage = self.model.stages[s]
                            dead_idx = torch.where(~used[s])[0]
                            if len(dead_idx) > 0:
                                picks = torch.randint(0, residual.size(0), (len(dead_idx),), device=self.device)
                                stage.weight.data[dead_idx] = residual[picks]
                            dist = torch.cdist(residual, stage.weight, p=2.0)
                            tok = torch.argmin(dist, dim=1)
                            residual = residual - stage.weight[tok]
                        print(f"[RESTART] {dead_total} dead codes revived across {self.model.n_stages} stages")

                if avg_total < self.prev_avg_loss:
                    self.prev_avg_loss = avg_total
                    ckpt = {
                        "epoch": real_epoch,
                        "avg_loss": avg_total,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                    }
                    torch.save(ckpt, out_dir.joinpath("pvq_best_model.pth"))
        except KeyboardInterrupt:
            print("[INFO] Training blocked by user!")

    def load_model(self, model_path: str) -> None:
        if self.model is None:
            self.net_init()
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state"])
            except Exception:
                print("[WARN] Optimizer state incompatible")
        if "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
        self.start_epoch = state.get("epoch", 0) + 1
        self.prev_avg_loss = state.get("avg_loss", float("inf"))
