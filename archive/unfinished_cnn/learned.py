"""A learned inspector: a small 3D CNN that sees (MRI, mask) as two channels and predicts
'mask was corrupted'. Trained only on *online-generated* corruptions of the training cases.

Guards against over-fitting (only ~80 training cases):
  - split by case (train / val / test are disjoint cases)
  - corruptions are re-drawn every time a case is used (type, severity, direction, location)
  - flips, small joint shifts, intensity/noise jitter
  - every mask, clean or corrupted, goes through the same smoothing so the network cannot
    use "blocky morphology artefacts" as a shortcut
  - tiny network, dropout, weight decay, early stopping on validation AUROC
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

from .corrupt import CORRUPTIONS

TYPES = list(CORRUPTIONS)
CANVAS = (48, 64, 48)   # every case is zero-padded to this (largest case is 43x59x47)


def smooth_mask(mask, sigma=0.7):
    """Volume-preserving-ish smoothing applied to ALL masks (clean and corrupted)."""
    return ndi.gaussian_filter(mask.astype(np.float32), sigma) > 0.5


def pad_to_canvas(a):
    out = np.zeros(CANVAS, dtype=np.float32)
    s = a.shape
    out[:s[0], :s[1], :s[2]] = a
    return out


def make_input(img, mask):
    return np.stack([pad_to_canvas(img), pad_to_canvas(mask.astype(np.float32))])


def corrupt(mask, ctype, sev, rng):
    return CORRUPTIONS[ctype](mask, sev, rng)


def augment_pair(img, mask, rng):
    """Joint geometric + photometric augmentation (image and mask stay aligned)."""
    for ax in range(3):
        if rng.random() < 0.5:
            img, mask = np.flip(img, ax), np.flip(mask, ax)
    sh = rng.integers(-3, 4, 3)
    img, mask = np.roll(img, tuple(sh), (0, 1, 2)), np.roll(mask, tuple(sh), (0, 1, 2))
    img = img * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05)
    img = img + rng.normal(0, 0.02, img.shape)
    return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(mask)


class Net(nn.Module):
    def __init__(self, width=8, p_drop=0.3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(2, o), nn.ReLU(),
                                 nn.MaxPool3d(2))
        self.f = nn.Sequential(block(2, width), block(width, 2 * width), block(2 * width, 4 * width))
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(8 * width, 1))

    def forward(self, x):
        h = self.f(x)
        h = torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], 1)
        return self.head(h).squeeze(1)


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sample_batch(cases, train_types, bs, rng, shuffle_labels=False):
    xs, ys = [], []
    for _ in range(bs):
        img, mask = cases[int(rng.integers(len(cases)))]
        img, mask = augment_pair(img, mask, rng)
        y = float(rng.random() < 0.5)
        if y:
            ctype = train_types[int(rng.integers(len(train_types)))]
            mask = corrupt(mask, ctype, int(rng.integers(1, 4)), rng)
        xs.append(make_input(img, smooth_mask(mask)))
        ys.append(y)
    ys = np.array(ys, dtype=np.float32)
    if shuffle_labels:                      # control: labels carry no information
        ys = rng.permutation(ys)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(ys)


@torch.no_grad()
def predict(model, inputs, device, bs=32):
    model.eval()
    out = []
    for i in range(0, len(inputs), bs):
        out.append(model(torch.from_numpy(np.stack(inputs[i:i + bs])).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def make_val_set(cases, train_types, seed=123):
    """Fixed validation items (disjoint cases): half clean, half corrupted with the training types."""
    rng = np.random.default_rng(seed)
    inputs, ys = [], []
    for k, (img, mask) in enumerate(cases):
        inputs.append(make_input(img, smooth_mask(mask))); ys.append(0)
        for ctype in train_types:
            m2 = corrupt(mask, ctype, int(rng.integers(2, 4)), rng)
            inputs.append(make_input(img, smooth_mask(m2))); ys.append(1)
    return inputs, np.array(ys)


def train(train_cases, val_cases, train_types, seed, epochs=40, steps=12, bs=24,
          shuffle_labels=False, device=None, verbose=False):
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    val_x, val_y = make_val_set(val_cases, train_types)
    best, best_state, since = -1, None, 0
    for ep in range(epochs):
        model.train()
        for _ in range(steps):
            x, y = sample_batch(train_cases, train_types, bs, rng, shuffle_labels)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        auc = roc_auc_score(val_y, predict(model, val_x, device))
        if verbose:
            print(f"    epoch {ep + 1:2d} loss {loss.item():.3f} val AUROC {auc:.3f}")
        if auc > best:
            best, since = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= 8:
                break
    model.load_state_dict(best_state)
    return model, best
