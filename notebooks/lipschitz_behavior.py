# %% [1] Setup and Data Generation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Global Plotting Config
plt.rcParams["font.family"] = "monospace"
PRIMARY = np.array([166, 0, 0]) / 255
CONTRARY = np.array([0, 166, 166]) / 255
NEUTRAL_MEDIUM_GREY = np.array([128, 128, 128]) / 255
NEUTRAL_DARK_GREY = np.array([64, 64, 64]) / 255

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def target_function(x):
    """Target function: y = sin(1/x) with clipping for stability."""
    x = np.sign(x) * np.clip(np.abs(x), 1e-4, 1)
    return np.sin(1 / x)


def get_data(n=2048):
    c = np.linspace(-1, 1, n)
    y = target_function(c)
    return (
        torch.FloatTensor(c).view(-1, 1).to(device),
        torch.FloatTensor(y).view(-1, 1).to(device),
    )


# %% [2] Model Definition
class SineRegressor(nn.Module):
    def __init__(self, hidden=256, layers=3):
        super().__init__()
        # Input is 3D: [current_x, conditioning_label_c, time_t]
        net = [nn.Linear(3, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            net.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x, c, t):
        return self.net(torch.cat([x, c, t], dim=-1))


# %% [3] Training Logic
def train_model(model, c_data, y_gt, loss_type, epochs=3000):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        batch_sz = c_data.shape[0]

        if loss_type == "regression":
            # Direct mapping: x_in=0, t=0 -> output=y
            x_in = torch.zeros_like(y_gt)
            t = torch.zeros_like(y_gt)
        elif loss_type == "flow":
            # Flow Matching: Interpolate between Gaussian noise and ground truth
            t = torch.rand((batch_sz, 1), device=device)
            z = torch.randn_like(y_gt)
            x_in = (1 - t) * z + t * y_gt
        elif loss_type == "mip":
            # Multi-Input Prediction: mix of noise-free and partial-noise inputs
            mask = torch.rand((batch_sz, 1), device=device) < 0.5
            x_in = torch.where(
                mask, torch.zeros_like(y_gt), 0.1 * torch.randn_like(y_gt) + y_gt
            )
            t = torch.where(mask, torch.zeros_like(y_gt), 0.9 * torch.ones_like(y_gt))

        pred = model(x_in, c_data, t)
        loss = criterion(pred, y_gt)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"[{loss_type}] Epoch {epoch} Loss: {loss.item():.6f}")
    return model


# %% [4] Evaluation/Inference Logic
@torch.no_grad()
def sample(model, c, loss_type, mode="deterministic", steps=10):
    model.eval()
    batch_sz = c.shape[0]

    if loss_type == "regression":
        return model(
            torch.zeros(batch_sz, 1, device=device),
            c,
            torch.zeros(batch_sz, 1, device=device),
        )

    elif loss_type == "flow":
        # Integration loop
        xt = (
            torch.zeros(batch_sz, 1, device=device)
            if mode == "deterministic"
            else torch.randn(batch_sz, 1, device=device)
        )
        ts = np.linspace(0, 1, steps + 1)
        for i in range(steps):
            s, t = ts[i], ts[i + 1]
            s_tensor = torch.full((batch_sz, 1), s, device=device)
            y_pred = model(xt, c, s_tensor)
            xt = (1 - t) / (1 - s) * xt + (t - s) / (1 - s) * y_pred
        return xt

    elif loss_type == "mip":
        # Two-step inference
        x0 = model(
            torch.zeros(batch_sz, 1, device=device),
            c,
            torch.zeros(batch_sz, 1, device=device),
        )
        return model(x0, c, 0.9 * torch.ones(batch_sz, 1, device=device))


# %% [5] Execution: Train all models
c_train, y_train = get_data(2048)

print("Training Regression Model...")
model_reg = train_model(SineRegressor().to(device), c_train, y_train, "regression")

print("\nTraining Flow Model...")
model_flow = train_model(SineRegressor().to(device), c_train, y_train, "flow")

print("\nTraining MIP Model...")
model_mip = train_model(SineRegressor().to(device), c_train, y_train, "mip")

# %% [6] Inference: Generate predictions
c_test, y_test = get_data(4096)  # Higher resolution for testing

y_pred_reg = sample(model_reg, c_test, "regression").cpu().numpy()
y_pred_flow_det = sample(model_flow, c_test, "flow", mode="deterministic").cpu().numpy()
y_pred_flow_sto = sample(model_flow, c_test, "flow", mode="stochastic").cpu().numpy()
y_pred_mip = sample(model_mip, c_test, "mip").cpu().numpy()

x_cpu = c_test.cpu().numpy()
gt_cpu = y_test.cpu().numpy()

# %% [7] Plotting
fig, axs = plt.subplots(1, 5, figsize=(15, 3))

titles = [
    r"Ground Truth $y=\sin(1/x)$",
    "Regression",
    r"Flow ($z=0$)",
    r"Flow ($z \sim \mathcal{N}$)",
    "MIP",
]
results = [gt_cpu, y_pred_reg, y_pred_flow_det, y_pred_flow_sto, y_pred_mip]
colors = [NEUTRAL_MEDIUM_GREY, NEUTRAL_DARK_GREY, CONTRARY, CONTRARY, PRIMARY]

for i, ax in enumerate(axs):
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")

    if i == 0:
        ax.plot(x_cpu, results[i], color=colors[i], linewidth=0.8)
        ax.set_ylabel("y")
    else:
        ax.scatter(x_cpu, results[i], color=colors[i], s=1, alpha=0.7)

    ax.set_title(titles[i], fontsize=10)

plt.tight_layout()
plt.show()

# %%
