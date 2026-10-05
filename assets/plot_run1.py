"""Plot run 1's loss and HellaSwag curves from run1_log.txt.

Writes assets/run1_light.png and assets/run1_dark.png (the README picks one per theme).
Needs matplotlib, which is not a training dependency: pip install matplotlib
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKENS_PER_STEP = 524288

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e8e7e3",
                  train="#2a78d6", val="#eb6834"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#383835",
                  train="#3987e5", val="#d95926"),
}

def parse(path):
    train, val, hs = [], [], []
    step = 0
    for line in open(path):
        if m := re.match(r"step (\d+), loss: ([\d.]+)", line):
            train.append((int(m[1]), float(m[2])))
        elif m := re.match(r"step (\d+), validation loss: ([\d.]+)", line):
            step = int(m[1])
            val.append((step, float(m[2])))
        elif m := re.match(r"hellaswag: \d+/\d+ = ([\d.]+)%", line):
            hs.append((step, float(m[1])))   # logged right after that step's val loss
    return train, val, hs

def ema(ys, alpha=0.02):
    out, s = [], ys[0]
    for y in ys:
        s = alpha * y + (1 - alpha) * s
        out.append(s)
    return out

def btok(steps):
    return [s * TOKENS_PER_STEP / 1e9 for s in steps]

def plot(theme, train, val, hs, out):
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
    fig.patch.set_facecolor(c["surface"])

    for ax in (ax1, ax2):
        ax.set_facecolor(c["surface"])
        ax.grid(axis="y", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(c["ink2"])
        ax.tick_params(colors=c["ink2"], length=0, pad=6)
        ax.set_xlabel("tokens seen (billions)", color=c["ink2"])
        ax.set_xlim(0, 10.9)

    # loss: raw train faintly, smoothed train and val on top; clipped to show the late curve
    ts, tl = zip(*train)
    vs, vl = zip(*val)
    ax1.plot(btok(ts), tl, color=c["train"], alpha=0.15, linewidth=0.6)
    ax1.plot(btok(ts), ema(tl), color=c["train"], linewidth=2, label="train (smoothed)")
    ax1.plot(btok(vs), vl, color=c["val"], linewidth=2, label="validation")
    ax1.axhline(3.29, color=c["ink2"], linewidth=1, linestyle=(0, (4, 3)))
    ax1.text(10.85, 3.29 + 0.03, "GPT-2 124M ≈ 3.29", color=c["ink2"], ha="right", va="bottom", fontsize=10)
    ax1.annotate(f"{vl[-1]:.2f}", (btok([vs[-1]])[0], vl[-1]), xytext=(6, 0),
                 textcoords="offset points", ha="left", va="center", color=c["ink"], fontsize=10, fontweight="bold")
    ax1.set_ylim(2.75, 4.6)
    ax1.set_title("Loss", loc="left", color=c["ink"], fontweight="bold")
    leg = ax1.legend(frameon=False, loc="upper right", labelcolor=c["ink2"])

    # hellaswag with published reference points. Each label sits where the curve isn't:
    # GPT-3's below its line on the right (the curve stays above it after 8.5B tokens),
    # the lower two above theirs on the left (the curve passes them in the first 0.5B)
    hx, hy = zip(*hs)
    for y, label, right in [(33.7, "GPT-3 125M  33.7%", True), (29.6, "GPT-2 124M  29.6%", False),
                            (27.4, "untrained  27.4%", False)]:
        ax2.axhline(y, color=c["ink2"], linewidth=1, linestyle=(0, (4, 3)))
        if right:
            ax2.text(10.85, y - 0.12, label, color=c["ink2"], ha="right", va="top", fontsize=10)
        else:
            ax2.text(0.7, y + 0.12, label, color=c["ink2"], ha="left", va="bottom", fontsize=10)
    ax2.plot(btok(hx), hy, color=c["train"], linewidth=2, marker="o", markersize=5,
             markeredgecolor=c["surface"], markeredgewidth=1.5)
    ax2.annotate(f"{hy[-1]:.1f}%", (btok([hx[-1]])[0], hy[-1]), xytext=(7, 0),
                 textcoords="offset points", ha="left", va="center", color=c["ink"], fontsize=10, fontweight="bold")
    ax2.set_ylim(26.5, 36)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax2.set_title("HellaSwag accuracy (n=1000)", loc="left", color=c["ink"], fontweight="bold")

    fig.suptitle("Run 1: 153M params, 10B FineWeb-Edu tokens, 2x H100, 3.2 hours",
                 x=0.012, ha="left", color=c["ink"], fontsize=13, fontweight="bold")
    fig.text(0.012, 0.885, "Loss axis starts at 4.6; the first ~0.5B tokens fall from 11.0 off the top.",
             color=c["ink2"], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=3)
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)

if __name__ == "__main__":
    data = parse("run1_log.txt")
    for theme in THEMES:
        plot(theme, *data, f"assets/run1_{theme}.png")
