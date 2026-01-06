# ddpm_cats.py
import os, math, argparse, random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

# -----------------------------
# 工具：隨機種子
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# 時間位置嵌入（sinusoidal）
# -----------------------------
def sinusoidal_time_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

# -----------------------------
# UNet 組件 (跟原本一致，但支援動態 img_ch)
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim)
        self.pool   = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)  # 下採樣

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        skip = x
        x = self.pool(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up     = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)  # 上採樣
        self.block1 = ResidualBlock(out_ch+in_ch, out_ch, time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x

class SimpleUNet(nn.Module):
    def __init__(self, img_ch=3, base_ch=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(img_ch, base_ch, 3, padding=1)
        self.down1 = Down(base_ch, base_ch*2, time_dim)      # e.g., 64->32
        self.down2 = Down(base_ch*2, base_ch*4, time_dim)    # e.g., 32->16
        self.down3 = Down(base_ch*4, base_ch*8, time_dim)    # e.g., 16->8

        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_dim)
        self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_dim)

        self.up1  = Up(base_ch*8, base_ch*4, time_dim)       # 8->16
        self.up2  = Up(base_ch*4, base_ch*2, time_dim)       # 16->32
        self.up3  = Up(base_ch*2, base_ch, time_dim)         # 32->64
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_ch, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)

        x = self.bot1(x, t_emb)
        x = self.bot2(x, t_emb)

        x = self.up1(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up3(x, s1, t_emb)
        return self.out_conv(x)

# -----------------------------
# Diffusion 參數與流程
# -----------------------------
@dataclass
class DiffusionConfig:
    img_size: int = 64
    channels: int = 3
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

class DDPM:
    def __init__(self, cfg: DiffusionConfig, device):
        self.cfg = cfg
        self.device = device
        self.T = cfg.timesteps

        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, self.T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_ac * x0 + sqrt_om * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t):
        betas_t = self.betas[t].view(-1,1,1,1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1,1,1,1)
        sqrt_one_minus_ac_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)

        eps_theta = model(x, t)
        x0_pred = (x - eps_theta*betas_t / sqrt_one_minus_ac_t) * sqrt_recip_alpha_t

        posterior_var_t = self.posterior_variance[t].view(-1,1,1,1)
        mu = x0_pred
        if (t == 0).all():
            return mu
        noise = torch.randn_like(x)
        return mu + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model, n=64):
        model.eval()
        x = torch.randn(n, self.cfg.channels, self.cfg.img_size, self.cfg.img_size, device=self.device)
        for t in tqdm(reversed(range(self.T)), total=self.T, desc="Sampling"):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        x = x.clamp(-1, 1)
        return x
    
    @torch.no_grad()
    def sample_path(self, model, n=64):
        path = []
        model.eval()
        x = torch.randn(n, self.cfg.channels, self.cfg.img_size, self.cfg.img_size, device=self.device)
        path.append(x)
        for t in tqdm(reversed(range(self.T)), total=self.T, desc="Sampling"):
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
            path.append(x)
        x = x.clamp(-1, 1)
        path.append(x)
        return path

# -----------------------------
# 資料
# -----------------------------
def get_dataloader(data_dir="./data/borhanitrash/cat-dataset", img_size=64, batch_size=64, num_workers=4):
    assert img_size % 4 == 0, "img_size must be divisible by 4"
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),  # map to [-1,1]
    ])
    ds = datasets.ImageFolder(root=data_dir, transform=tfm)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in {data_dir}. Make sure dataset is organized as ImageFolder (class subfolders).")
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)

# -----------------------------
# 訓練
# -----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dl = get_dataloader(data_dir=args.data_dir, img_size=args.img_size,
                        batch_size=args.batch_size, num_workers=args.workers)

    cfg = DiffusionConfig(img_size=args.img_size, channels=args.channels,
                          timesteps=args.timesteps, beta_start=args.beta_start,
                          beta_end=args.beta_end)
    ddpm = DDPM(cfg, device)

    model = SimpleUNet(img_ch=cfg.channels, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.outdir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl, total=len(dl), desc=f"Epoch {epoch}/{args.epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)  # range should be [-1,1]
            b = x0.size(0)
            t = torch.randint(0, cfg.timesteps, (b,), device=device).long()

            with torch.cuda.amp.autocast(enabled=args.amp):
                x_t, noise = ddpm.q_sample(x0, t)
                pred_noise = model(x_t, t)
                loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 每個 epoch 取樣並存圖
        with torch.no_grad():
            samples = ddpm.sample(model, n=args.sample_n)
            grid = (samples + 1) / 2  # -> [0,1]
            save_path = os.path.join(args.gendir, f"sample/sample_epoch_{epoch}.png")
            utils.save_image(grid, save_path, nrow=int(math.sqrt(args.sample_n)))
            print(f"[Info] Saved samples to: {save_path}")

        # 儲存 checkpoint
        ckpt_path = os.path.join(args.outdir, f"ddpm_cats_epoch_{epoch}.pt")
        torch.save({
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "args": vars(args),
        }, ckpt_path)
        print(f"[Info] Saved checkpoint: {ckpt_path}")
        
def generate_path(model_path):
    model_data = torch.load(model_path, map_location="cpu")
    args = argparse.Namespace(**model_data["args"])
    cfg = argparse.Namespace(**model_data["cfg"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM(cfg, device)
    model = SimpleUNet(img_ch=cfg.channels, base_ch=args.base_ch, time_dim=args.time_dim)
    model.load_state_dict(model_data["model"])
    model.to(device)
    os.makedirs(args.gendir, exist_ok=True)
    with torch.no_grad():
        samples_path = ddpm.sample_path(model, n=args.sample_n)
        for i,samples in enumerate(samples_path):
            grid = (samples + 1) / 2
            save_path = os.path.join(args.gendir, f"images_sample_step_{i}.png")
            utils.save_image(grid, save_path, nrow=int(math.sqrt(args.sample_n)))
            print(f"[Info] Saved samples to: {save_path}")

# -----------------------------
# 主程式
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DDPM for Cat Dataset")
    p.add_argument("--data_dir", type=str, default="./data/cats",
                   help="path to root folder of dataset (ImageFolder structure)")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--base_ch", type=int, default=128)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--sample_n", type=int, default=16)
    p.add_argument("--outdir", type=str, default="./runs_ddpm_cats/ckpt")
    p.add_argument("--gendir", type=str, default="./runs_ddpm_cats/generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--img_size", type=int, default=64, help="image size (must be divisible by 4)")
    p.add_argument("--channels", type=int, default=3)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 如果你要訓練，執行 train(args)
    train(args)
    # 若只要產樣本或回放路徑，可以呼叫 generate_path("path_to_ckpt")
