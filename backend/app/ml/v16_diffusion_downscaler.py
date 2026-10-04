from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# VATAVARAN V16
# Conditional Diffusion Downscaler Prototype
#
# IMPORTANT:
# - Current V8 source grid = 30x30 at ~1 degree spacing.
# - This prototype produces a 5 km high-resolution probabilistic field.
# - There is NO paired 12 km/5 km ground-truth dataset in the current repo.
# - Therefore training targets are synthetic physics-guided fine-scale fields.
# - This is NOT claimed as a validated operational 12 km -> 5 km model.
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

V8_PATH = DATA_DIR / "v8_extreme_precip_grid_forecasts_2025.npz"
CHECKPOINT_PATH = MODEL_DIR / "vatavaran_v16_diffusion_downscaler.pt"

GRID = 30
COARSE_CROP = 5
DEFAULT_FINE_SIZE = 41
DEFAULT_DIFFUSION_STEPS = 40

# Normalisation scales:
# temperature °C, humidity %, wind km/h, precipitation mm
CHANNEL_SCALES = np.array(
    [50.0, 100.0, 100.0, 200.0],
    dtype=np.float32,
)

CHANNEL_NAMES = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
]

HAZARDS = [
    "EXTREME_HEAT",
    "HIGH_WIND",
    "EXTREME_PRECIPITATION",
]

HAZARD_TO_ID = {name: i for i, name in enumerate(HAZARDS)}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_hazard(hazard: str) -> str:
    hazard = str(hazard or "").strip().upper()

    aliases = {
        "HEAT": "EXTREME_HEAT",
        "HIGH_WIND": "HIGH_WIND",
        "WIND": "HIGH_WIND",
        "EXTREME_PRECIP": "EXTREME_PRECIPITATION",
        "PRECIPITATION": "EXTREME_PRECIPITATION",
        "RAIN": "EXTREME_PRECIPITATION",
    }

    hazard = aliases.get(hazard, hazard)

    if hazard not in HAZARD_TO_ID:
        hazard = "EXTREME_PRECIPITATION"

    return hazard


def normalize_array(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / CHANNEL_SCALES.reshape(4, 1, 1)


def denormalize_tensor(x: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(
        CHANNEL_SCALES,
        dtype=x.dtype,
        device=x.device,
    ).view(1, 4, 1, 1)

    return x * scale


def load_v8() -> dict:
    if not V8_PATH.exists():
        raise FileNotFoundError(
            f"V8 dataset not found: {V8_PATH}"
        )

    with np.load(V8_PATH, allow_pickle=True) as data:
        return {
            "timestamps": data["timestamps"],
            "target_timestamps": data["target_timestamps"],
            "horizons": data["horizons"],
            "latitudes": data["latitudes"],
            "longitudes": data["longitudes"],
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "wind": data["wind"],
            "precipitation": data["precipitation"],
            "occurrence_probability": data["occurrence_probability"],
        }


def nearest_node(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lat: float,
    lon: float,
) -> int:
    distance = (latitudes - lat) ** 2 + (longitudes - lon) ** 2
    return int(np.argmin(distance))


def extract_coarse_patch(
    data: dict,
    timestamp_index: int,
    horizon_index: int,
    center_lat: float,
    center_lon: float,
    crop_size: int = COARSE_CROP,
) -> np.ndarray:
    node = nearest_node(
        data["latitudes"],
        data["longitudes"],
        center_lat,
        center_lon,
    )

    row = node // GRID
    col = node % GRID

    half = crop_size // 2

    # Variables from the V8 forecast field.
    fields = [
        data["temperature"][timestamp_index, horizon_index],
        data["humidity"][timestamp_index, horizon_index],
        data["wind"][timestamp_index, horizon_index],
        data["precipitation"][timestamp_index, horizon_index],
    ]

    patch_channels = []

    for flat_field in fields:
        grid = flat_field.reshape(GRID, GRID)

        padded = np.pad(
            grid,
            ((half, half), (half, half)),
            mode="edge",
        )

        patch = padded[
            row : row + crop_size,
            col : col + crop_size,
        ]

        patch_channels.append(
            patch.astype(np.float32)
        )

    return np.stack(
        patch_channels,
        axis=0,
    )


def gaussian_field(
    size: int,
    sigma: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, size)

    yy, xx = np.meshgrid(
        coords,
        coords,
    )

    distance = (
        (xx - shift_x) ** 2
        + (yy - shift_y) ** 2
    )

    return np.exp(
        -distance
        / max(2.0 * sigma * sigma, 1e-6)
    ).astype(np.float32)


def correlated_noise(
    size: int,
    amplitude: float,
) -> np.ndarray:
    coarse = np.random.normal(
        0.0,
        1.0,
        size=(8, 8),
    ).astype(np.float32)

    tensor = torch.from_numpy(
        coarse
    ).unsqueeze(0).unsqueeze(0)

    smooth = F.interpolate(
        tensor,
        size=(size, size),
        mode="bicubic",
        align_corners=True,
    )[0, 0]

    smooth = smooth.numpy()

    std = float(
        np.std(smooth)
    )

    if std > 1e-6:
        smooth = smooth / std

    return smooth * amplitude


def synthesize_fine_target(
    coarse_orig: np.ndarray,
    hazard: str,
    fine_size: int,
) -> np.ndarray:
    """
    Produce a physically plausible synthetic fine-scale target.

    This is ONLY a prototype training target because paired fine-resolution
    observations are absent from the current repository.
    """

    coarse_tensor = torch.from_numpy(
        coarse_orig
    ).unsqueeze(0)

    base = F.interpolate(
        coarse_tensor,
        size=(fine_size, fine_size),
        mode="bicubic",
        align_corners=True,
    )[0].numpy()

    temperature = base[0].copy()
    humidity = base[1].copy()
    wind = np.maximum(base[2].copy(), 0.0)
    precipitation = np.maximum(
        base[3].copy(),
        0.0,
    )

    core = gaussian_field(
        fine_size,
        sigma=0.30,
        shift_x=np.random.uniform(-0.18, 0.18),
        shift_y=np.random.uniform(-0.18, 0.18),
    )

    broad = gaussian_field(
        fine_size,
        sigma=0.55,
    )

    n_temp = correlated_noise(
        fine_size,
        amplitude=0.45,
    )

    n_humidity = correlated_noise(
        fine_size,
        amplitude=2.5,
    )

    n_wind = correlated_noise(
        fine_size,
        amplitude=1.8,
    )

    n_precip = correlated_noise(
        fine_size,
        amplitude=2.0,
    )

    hazard = safe_hazard(hazard)

    if hazard == "EXTREME_HEAT":
        temperature += (
            3.5 * core
            + 0.8 * broad
            + n_temp
        )

        humidity -= (
            8.0 * core
            - n_humidity
        )

        precipitation *= (
            1.0 - 0.65 * core
        )

        wind += (
            2.0 * broad
            + n_wind
        )

    elif hazard == "HIGH_WIND":
        wind += (
            18.0 * core
            + 5.0 * broad
            + n_wind
        )

        temperature += (
            0.5 * broad
            + n_temp
        )

        humidity += (
            3.5 * core
            + n_humidity
        )

        precipitation += (
            8.0 * core
            + np.maximum(n_precip, 0.0)
        )

    else:
        precipitation += (
            45.0 * core
            + 12.0 * broad
            + np.maximum(n_precip, 0.0)
        )

        humidity += (
            10.0 * core
            + n_humidity
        )

        wind += (
            5.0 * core
            + n_wind
        )

        temperature -= (
            1.2 * core
        )

    humidity = np.clip(
        humidity,
        0.0,
        100.0,
    )

    wind = np.clip(
        wind,
        0.0,
        200.0,
    )

    precipitation = np.clip(
        precipitation,
        0.0,
        500.0,
    )

    temperature = np.clip(
        temperature,
        -80.0,
        60.0,
    )

    return np.stack(
        [
            temperature,
            humidity,
            wind,
            precipitation,
        ],
        axis=0,
    ).astype(np.float32)


class TimeEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:

        half = self.dim // 2

        frequency = torch.exp(
            torch.arange(
                half,
                device=t.device,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / max(half - 1, 1)
            )
        )

        angles = (
            t.float().unsqueeze(1)
            * frequency.unsqueeze(0)
        )

        emb = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )

        return self.mlp(emb)


class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        time_dim: int,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            4,
            channels,
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.time_proj = nn.Linear(
            time_dim,
            channels,
        )

        self.norm2 = nn.GroupNorm(
            4,
            channels,
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = h + self.time_proj(
            t_emb
        ).unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + residual


class ConditionalDiffusionUNet(nn.Module):

    def __init__(
        self,
        base_channels: int = 32,
    ):
        super().__init__()

        # noisy field: 4 channels
        # coarse condition: 4 channels
        # hazard conditioning: 3 channels
        input_channels = 11

        self.time = TimeEmbedding(
            base_channels
        )

        self.input_conv = nn.Conv2d(
            input_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.block1 = ResidualBlock(
            base_channels,
            base_channels,
        )

        self.block2 = ResidualBlock(
            base_channels,
            base_channels,
        )

        self.block3 = ResidualBlock(
            base_channels,
            base_channels,
        )

        self.output_norm = nn.GroupNorm(
            4,
            base_channels,
        )

        self.output_conv = nn.Conv2d(
            base_channels,
            4,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        noisy: torch.Tensor,
        condition: torch.Tensor,
        hazard_map: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat(
            [
                noisy,
                condition,
                hazard_map,
            ],
            dim=1,
        )

        t_emb = self.time(
            timestep
        )

        h = self.input_conv(x)
        h = self.block1(h, t_emb)
        h = self.block2(h, t_emb)
        h = self.block3(h, t_emb)

        h = self.output_norm(h)
        h = F.silu(h)

        return self.output_conv(h)


class DiffusionSchedule:

    def __init__(
        self,
        steps: int,
        device: torch.device,
    ):
        self.steps = int(steps)

        self.betas = torch.linspace(
            1e-4,
            0.02,
            self.steps,
            device=device,
        )

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )

        self.alpha_bars_prev = torch.cat(
            [
                torch.ones(
                    1,
                    device=device,
                ),
                self.alpha_bars[:-1],
            ]
        )

        self.posterior_variance = (
            self.betas
            * (
                1.0
                - self.alpha_bars_prev
            )
            / (
                1.0
                - self.alpha_bars
            )
        ).clamp_min(1e-12)


def hazard_map(
    hazard: torch.Tensor,
    size: int,
) -> torch.Tensor:

    one_hot = F.one_hot(
        hazard.long(),
        num_classes=3,
    ).float()

    return one_hot.unsqueeze(-1).unsqueeze(-1).expand(
        -1,
        -1,
        size,
        size,
    )


def physics_loss(
    predicted_norm: torch.Tensor,
    coarse_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Physics-guided regularisation:

    1. valid atmospheric ranges
    2. precipitation non-negativity
    3. humidity/precipitation consistency
    4. local precipitation conservation
    5. spatial smoothness
    """

    predicted = denormalize_tensor(
        predicted_norm
    )

    coarse = denormalize_tensor(
        coarse_norm
    )

    temperature = predicted[:, 0]
    humidity = predicted[:, 1]
    wind = predicted[:, 2]
    precipitation = predicted[:, 3]

    temp_penalty = (
        F.relu(-80.0 - temperature)
        + F.relu(temperature - 60.0)
    ).mean()

    humidity_penalty = (
        F.relu(-humidity)
        + F.relu(humidity - 100.0)
    ).mean()

    wind_penalty = (
        F.relu(-wind)
        + F.relu(wind - 200.0)
    ).mean()

    precip_penalty = (
        F.relu(-precipitation)
        + F.relu(precipitation - 500.0)
    ).mean()

    # High precipitation should be supported by sufficiently moist air.
    moisture_threshold = (
        40.0
        + 0.35 * humidity
    )

    moisture_penalty = F.relu(
        precipitation
        - 1.6 * moisture_threshold
    ).mean()

    # Preserve coarse-domain precipitation mean.
    coarse_precip_mean = coarse[:, 3].mean(
        dim=(1, 2)
    )

    fine_precip_mean = precipitation.mean(
        dim=(1, 2)
    )

    conservation_penalty = (
        fine_precip_mean
        - coarse_precip_mean
    ).abs().mean()

    # Spatial smoothness without destroying local extremes.
    grad_x = (
        precipitation[:, :, 1:]
        - precipitation[:, :, :-1]
    )

    grad_y = (
        precipitation[:, 1:, :]
        - precipitation[:, :-1, :]
    )

    smoothness_penalty = (
        grad_x.abs().mean()
        + grad_y.abs().mean()
    ) * 0.02

    total = (
        0.01 * temp_penalty
        + 0.01 * humidity_penalty
        + 0.01 * wind_penalty
        + 0.02 * precip_penalty
        + 0.02 * moisture_penalty
        + 0.03 * conservation_penalty
        + smoothness_penalty
    )

    return total


def make_batch(
    data: dict,
    indices: list[tuple[int, int]],
    fine_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    conditions = []
    targets = []
    hazards = []

    for timestamp_index, horizon_index in indices:

        center_lat = float(
            np.random.uniform(
                10.0,
                35.0,
            )
        )

        center_lon = float(
            np.random.uniform(
                70.0,
                95.0,
            )
        )

        hazard = random.choice(
            HAZARDS
        )

        coarse = extract_coarse_patch(
            data,
            timestamp_index,
            horizon_index,
            center_lat,
            center_lon,
        )

        target = synthesize_fine_target(
            coarse,
            hazard,
            fine_size,
        )

        coarse_tensor = torch.from_numpy(
            coarse
        ).unsqueeze(0)

        coarse_up = F.interpolate(
            coarse_tensor,
            size=(fine_size, fine_size),
            mode="bicubic",
            align_corners=True,
        )[0].numpy()

        conditions.append(
            normalize_array(
                coarse_up
            )
        )

        targets.append(
            normalize_array(
                target
            )
        )

        hazards.append(
            HAZARD_TO_ID[hazard]
        )

    condition_tensor = torch.from_numpy(
        np.stack(conditions)
    ).to(device)

    target_tensor = torch.from_numpy(
        np.stack(targets)
    ).to(device)

    hazard_tensor = torch.tensor(
        hazards,
        dtype=torch.long,
        device=device,
    )

    return (
        condition_tensor,
        target_tensor,
        hazard_tensor,
    )


def q_sample(
    x0: torch.Tensor,
    timestep: torch.Tensor,
    schedule: DiffusionSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:

    noise = torch.randn_like(
        x0
    )

    alpha_bar = (
        schedule.alpha_bars[timestep]
        .view(-1, 1, 1, 1)
    )

    noisy = (
        torch.sqrt(alpha_bar)
        * x0
        + torch.sqrt(
            1.0 - alpha_bar
        )
        * noise
    )

    return noisy, noise


def project_physics(
    x_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient-safe physical projection.

    No in-place tensor mutation is used, so autograd remains valid.
    """

    x = denormalize_tensor(
        x_norm
    )

    temperature = x[:, 0].clamp(
        -80.0,
        60.0,
    )

    humidity = x[:, 1].clamp(
        0.0,
        100.0,
    )

    wind = x[:, 2].clamp(
        0.0,
        200.0,
    )

    precipitation = x[:, 3].clamp(
        0.0,
        500.0,
    )

    projected = torch.stack(
        [
            temperature,
            humidity,
            wind,
            precipitation,
        ],
        dim=1,
    )

    scales = torch.tensor(
        CHANNEL_SCALES,
        device=projected.device,
        dtype=projected.dtype,
    ).view(1, 4, 1, 1)

    return projected / scales


def train_model(
    epochs: int = 3,
    samples_per_epoch: int = 192,
    batch_size: int = 4,
    fine_size: int = DEFAULT_FINE_SIZE,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    lr: float = 2e-4,
    device_name: str = "cpu",
) -> Path:

    set_seed(42)

    device = torch.device(
        device_name
    )

    data = load_v8()

    model = ConditionalDiffusionUNet(
        base_channels=32
    ).to(device)

    schedule = DiffusionSchedule(
        diffusion_steps,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    model.train()

    total_cases = (
        len(data["timestamps"]),
        len(data["horizons"]),
    )

    print("=" * 100)
    print(
        "VATAVARAN V16 - CONDITIONAL DIFFUSION TRAINING"
    )
    print("=" * 100)

    print(
        f"Device              : {device}"
    )

    print(
        f"V8 timestamps       : {total_cases[0]}"
    )

    print(
        f"V8 horizons         : {total_cases[1]}"
    )

    print(
        f"Fine grid           : {fine_size} x {fine_size}"
    )

    print(
        f"Diffusion steps     : {diffusion_steps}"
    )

    print(
        "Training target      : synthetic physics-guided fine field"
    )

    print(
        "Ground-truth 5 km    : unavailable"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        running = 0.0

        for start in range(
            0,
            samples_per_epoch,
            batch_size,
        ):

            indices = []

            for _ in range(
                min(
                    batch_size,
                    samples_per_epoch - start,
                )
            ):
                indices.append(
                    (
                        random.randrange(
                            len(data["timestamps"])
                        ),
                        random.randrange(
                            len(data["horizons"])
                        ),
                    )
                )

            condition, target, hazard = make_batch(
                data,
                indices,
                fine_size,
                device,
            )

            timestep = torch.randint(
                0,
                diffusion_steps,
                (target.shape[0],),
                device=device,
            )

            noisy, noise = q_sample(
                target,
                timestep,
                schedule,
            )

            hmap = hazard_map(
                hazard,
                fine_size,
            )

            predicted_noise = model(
                noisy,
                condition,
                hmap,
                timestep,
            )

            diffusion_error = F.mse_loss(
                predicted_noise,
                noise,
            )

            alpha_bar = (
                schedule.alpha_bars[timestep]
                .view(-1, 1, 1, 1)
            )

            predicted_x0 = (
                noisy
                - torch.sqrt(
                    1.0 - alpha_bar
                )
                * predicted_noise
            ) / torch.sqrt(
                alpha_bar
            )

            predicted_x0 = project_physics(
                predicted_x0
            )

            physics_error = physics_loss(
                predicted_x0,
                condition,
            )

            loss = (
                diffusion_error
                + 0.15 * physics_error
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running += float(
                loss.detach().cpu()
            )

        mean_loss = (
            running
            / max(
                1,
                math.ceil(
                    samples_per_epoch
                    / batch_size
                ),
            )
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"loss={mean_loss:.6f}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "fine_size": fine_size,
            "diffusion_steps": diffusion_steps,
            "channel_scales": CHANNEL_SCALES,
            "channels": CHANNEL_NAMES,
            "hazards": HAZARDS,
            "base_channels": 32,
            "training_mode": (
                "synthetic_physics_guided_pretraining"
            ),
        },
        CHECKPOINT_PATH,
    )

    print(
        f"\nCheckpoint saved: {CHECKPOINT_PATH}"
    )

    return CHECKPOINT_PATH


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    ConditionalDiffusionUNet,
    int,
    DiffusionSchedule,
]:

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    fine_size = int(
        checkpoint["fine_size"]
    )

    diffusion_steps = int(
        checkpoint["diffusion_steps"]
    )

    model = ConditionalDiffusionUNet(
        base_channels=int(
            checkpoint.get(
                "base_channels",
                32,
            )
        )
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    schedule = DiffusionSchedule(
        diffusion_steps,
        device,
    )

    return (
        model,
        fine_size,
        schedule,
    )


@torch.no_grad()

def condition_anchored_projection(
    x_norm: torch.Tensor,
    condition_norm: torch.Tensor,
    hazard_id: int,
) -> torch.Tensor:
    """
    Keep the generated fine-scale field physically anchored to the
    coarse V8 atmospheric state.

    This is intentionally conservative because the repository does not
    contain paired 12 km / 5 km observations.
    """

    x = denormalize_tensor(
        x_norm
    )

    condition = denormalize_tensor(
        condition_norm
    )

    delta = x - condition

    # Conservative perturbation envelopes.
    temp_delta = 8.0
    humidity_delta = 20.0
    wind_delta = 30.0

    # Allow a stronger local precipitation enhancement for precipitation
    # hazards, but keep it tied to the coarse precipitation amount.
    coarse_precip = condition[:, 3]

    if int(hazard_id) == HAZARD_TO_ID["EXTREME_PRECIPITATION"]:
        precip_high = torch.maximum(
            torch.full_like(
                coarse_precip,
                15.0,
            ),
            coarse_precip * 25.0 + 10.0,
        )
    else:
        precip_high = torch.maximum(
            torch.full_like(
                coarse_precip,
                5.0,
            ),
            coarse_precip * 10.0 + 5.0,
        )

    delta = torch.stack(
        [
            delta[:, 0].clamp(
                -temp_delta,
                temp_delta,
            ),
            delta[:, 1].clamp(
                -humidity_delta,
                humidity_delta,
            ),
            delta[:, 2].clamp(
                -wind_delta,
                wind_delta,
            ),
            torch.minimum(
                torch.maximum(
                    delta[:, 3],
                    -torch.minimum(
                        coarse_precip,
                        torch.full_like(
                            coarse_precip,
                            2.0,
                        ),
                    ),
                ),
                precip_high,
            ),
        ],
        dim=1,
    )

    anchored = condition + delta

    anchored = torch.stack(
        [
            anchored[:, 0].clamp(
                -80.0,
                60.0,
            ),
            anchored[:, 1].clamp(
                0.0,
                100.0,
            ),
            anchored[:, 2].clamp(
                0.0,
                200.0,
            ),
            anchored[:, 3].clamp(
                0.0,
                500.0,
            ),
        ],
        dim=1,
    )

    scales = torch.tensor(
        CHANNEL_SCALES,
        device=anchored.device,
        dtype=anchored.dtype,
    ).view(1, 4, 1, 1)

    return anchored / scales


def sample_ensemble(
    model: ConditionalDiffusionUNet,
    schedule: DiffusionSchedule,
    condition: torch.Tensor,
    hazard_id: int,
    ensemble_size: int,
) -> torch.Tensor:
    """
    Generate a stochastic ensemble and repeatedly anchor the result to
    the coarse V8 atmospheric state.
    """

    batch_size = ensemble_size
    size = condition.shape[-1]

    condition = condition.repeat(
        batch_size,
        1,
        1,
        1,
    )

    hazard = torch.full(
        (batch_size,),
        int(hazard_id),
        dtype=torch.long,
        device=condition.device,
    )

    hmap = hazard_map(
        hazard,
        size,
    )

    x = torch.randn(
        batch_size,
        4,
        size,
        size,
        device=condition.device,
    )

    for step in reversed(
        range(
            schedule.steps
        )
    ):

        timestep = torch.full(
            (batch_size,),
            step,
            dtype=torch.long,
            device=condition.device,
        )

        predicted_noise = model(
            x,
            condition,
            hmap,
            timestep,
        )

        alpha = schedule.alphas[
            step
        ]

        alpha_bar = schedule.alpha_bars[
            step
        ]

        beta = schedule.betas[
            step
        ]

        predicted_x0 = (
            x
            - torch.sqrt(
                1.0 - alpha_bar
            )
            * predicted_noise
        ) / torch.sqrt(
            alpha_bar
        )

        predicted_x0 = project_physics(
            predicted_x0
        )

        predicted_x0 = condition_anchored_projection(
            predicted_x0,
            condition,
            hazard_id,
        )

        if step > 0:

            posterior_mean = (
                1.0
                / torch.sqrt(alpha)
            ) * (
                x
                - (
                    beta
                    / torch.sqrt(
                        1.0 - alpha_bar
                    )
                )
                * predicted_noise
            )

            posterior_variance = (
                schedule.posterior_variance[
                    step
                ]
            )

            noise = torch.randn_like(
                x
            )

            x = (
                posterior_mean
                + torch.sqrt(
                    posterior_variance
                )
                * noise
            )

            x = project_physics(
                x
            )

            x = condition_anchored_projection(
                x,
                condition,
                hazard_id,
            )

        else:
            x = predicted_x0

    return condition_anchored_projection(
        x,
        condition,
        hazard_id,
    )


def percentiles(

    samples: torch.Tensor,
) -> dict[str, np.ndarray]:

    array = (
        denormalize_tensor(
            samples
        )
        .detach()
        .cpu()
        .numpy()
    )

    return {
        "p10": np.percentile(
            array,
            10,
            axis=0,
        ),
        "p50": np.percentile(
            array,
            50,
            axis=0,
        ),
        "p90": np.percentile(
            array,
            90,
            axis=0,
        ),
        "p99": np.percentile(
            array,
            99,
            axis=0,
        ),
    }


def build_geojson(
    percentile_fields: dict[str, np.ndarray],
    center_lat: float,
    center_lon: float,
    fine_size: int,
    resolution_km: float = 5.0,
) -> list[dict]:

    half = (
        fine_size - 1
    ) / 2.0

    lat_step = (
        resolution_km
        / 111.0
    )

    cos_lat = max(
        math.cos(
            math.radians(
                center_lat
            )
        ),
        0.20,
    )

    lon_step = (
        resolution_km
        / (
            111.0
            * cos_lat
        )
    )

    p10 = percentile_fields["p10"]
    p50 = percentile_fields["p50"]
    p90 = percentile_fields["p90"]
    p99 = percentile_fields["p99"]

    features = []

    for r in range(
        fine_size
    ):
        for c in range(
            fine_size
        ):

            lat = (
                center_lat
                + (half - r)
                * lat_step
            )

            lon = (
                center_lon
                + (c - half)
                * lon_step
            )

            properties = {
                "temperature_p10_c": round(
                    float(p10[0, r, c]),
                    3,
                ),
                "temperature_p50_c": round(
                    float(p50[0, r, c]),
                    3,
                ),
                "temperature_p90_c": round(
                    float(p90[0, r, c]),
                    3,
                ),
                "temperature_p99_c": round(
                    float(p99[0, r, c]),
                    3,
                ),
                "humidity_p50_pct": round(
                    float(p50[1, r, c]),
                    3,
                ),
                "wind_p50_kmh": round(
                    float(p50[2, r, c]),
                    3,
                ),
                "precip_p10_mm": round(
                    float(p10[3, r, c]),
                    3,
                ),
                "precip_p50_mm": round(
                    float(p50[3, r, c]),
                    3,
                ),
                "precip_p90_mm": round(
                    float(p90[3, r, c]),
                    3,
                ),
                "precip_p99_mm": round(
                    float(p99[3, r, c]),
                    3,
                ),
                "resolution_km": resolution_km,
                "source": (
                    "VATAVARAN V16 conditional "
                    "diffusion prototype"
                ),
                "training_mode": (
                    "synthetic physics-guided "
                    "pretraining"
                ),
                "true_12km_to_5km": False,
            }

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            round(
                                float(lon),
                                6,
                            ),
                            round(
                                float(lat),
                                6,
                            ),
                        ],
                    },
                    "properties": properties,
                }
            )

    return features



def generate_from_coarse_patch(
    coarse_field: np.ndarray,
    center_lat: float,
    center_lon: float,
    hazard: str,
    ensemble_size: int = 10,
    device_name: str = "cpu",
) -> dict:
    """
    Run V16 conditional diffusion from an already extracted V8 coarse patch.

    coarse_field shape:
        (4, 5, 5)
        channels = temperature, humidity, wind, precipitation

    The current repository does not contain paired 12 km/5 km observations,
    so this remains a prototype using the checkpoint trained on synthetic
    physics-guided fine-scale targets.
    """

    coarse_field = np.asarray(
        coarse_field,
        dtype=np.float32,
    )

    if coarse_field.shape != (4, 5, 5):
        raise ValueError(
            "V16 expects coarse_field shape (4, 5, 5), "
            f"got {coarse_field.shape}"
        )

    if not np.isfinite(coarse_field).all():
        raise ValueError(
            "V16 coarse_field contains non-finite values."
        )

    hazard = safe_hazard(
        hazard
    )

    ensemble_size = max(
        1,
        min(
            int(ensemble_size),
            20,
        ),
    )

    device = torch.device(
        device_name
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"V16 checkpoint not found: {CHECKPOINT_PATH}"
        )

    model, fine_size, schedule = load_model(
        CHECKPOINT_PATH,
        device,
    )

    coarse_tensor = torch.from_numpy(
        coarse_field
    ).unsqueeze(0)

    coarse_up = F.interpolate(
        coarse_tensor,
        size=(fine_size, fine_size),
        mode="bicubic",
        align_corners=True,
    )

    condition = torch.from_numpy(
        normalize_array(
            coarse_up[0].numpy()
        )
    ).unsqueeze(0).to(device)

    samples = sample_ensemble(
        model,
        schedule,
        condition,
        HAZARD_TO_ID[hazard],
        ensemble_size,
    )

    p = percentiles(
        samples
    )

    features = build_geojson(
        p,
        center_lat,
        center_lon,
        fine_size,
        resolution_km=5.0,
    )

    return {
        "status": "success",
        "engine": "VATAVARAN V16 conditional diffusion",
        "resolution_km": 5.0,
        "fine_grid_size": fine_size,
        "coarse_source": "V8 30x30 ~1-degree grid",
        "hazard": hazard,
        "ensemble_samples": ensemble_size,
        "probabilistic_outputs": [
            "P10",
            "P50",
            "P90",
            "P99",
        ],
        "physics_constraint": True,
        "physics_training_loss": True,
        "true_12km_to_5km": False,
        "training_mode": (
            "synthetic physics-guided pretraining"
        ),
        "features": features,
    }


def demo_inference(
    center_lat: float,
    center_lon: float,
    hazard: str,
    timestamp_index: int = 100,
    horizon_index: int = 4,
    ensemble_size: int = 10,
    device_name: str = "cpu",
) -> dict:

    device = torch.device(
        device_name
    )

    data = load_v8()

    model, fine_size, schedule = load_model(
        CHECKPOINT_PATH,
        device,
    )

    coarse = extract_coarse_patch(
        data,
        timestamp_index,
        horizon_index,
        center_lat,
        center_lon,
    )

    coarse_tensor = torch.from_numpy(
        coarse
    ).unsqueeze(0)

    coarse_up = F.interpolate(
        coarse_tensor,
        size=(fine_size, fine_size),
        mode="bicubic",
        align_corners=True,
    )

    condition = torch.from_numpy(
        normalize_array(
            coarse_up[0].numpy()
        )
    ).unsqueeze(0).to(device)

    hazard = safe_hazard(
        hazard
    )

    samples = sample_ensemble(
        model,
        schedule,
        condition,
        HAZARD_TO_ID[hazard],
        ensemble_size,
    )

    p = percentiles(
        samples
    )

    features = build_geojson(
        p,
        center_lat,
        center_lon,
        fine_size,
    )

    return {
        "status": "success",
        "engine": "VATAVARAN V16 conditional diffusion",
        "resolution_km": 5.0,
        "coarse_source": "V8 30x30 ~1-degree grid",
        "hazard": hazard,
        "ensemble_samples": ensemble_size,
        "probabilistic_outputs": [
            "P10",
            "P50",
            "P90",
            "P99",
        ],
        "physics_constraint": True,
        "physics_training_loss": True,
        "true_12km_to_5km": False,
        "training_mode": (
            "synthetic physics-guided pretraining"
        ),
        "features": features,
    }


def smoke_test() -> None:

    set_seed(7)

    device = torch.device(
        "cpu"
    )

    model = ConditionalDiffusionUNet(
        base_channels=32
    ).to(device)

    schedule = DiffusionSchedule(
        8,
        device,
    )

    size = 17

    condition = torch.randn(
        1,
        4,
        size,
        size,
    )

    target = torch.randn(
        1,
        4,
        size,
        size,
    )

    hazard = torch.tensor(
        [2],
        dtype=torch.long,
    )

    timestep = torch.tensor(
        [4],
        dtype=torch.long,
    )

    noisy, noise = q_sample(
        target,
        timestep,
        schedule,
    )

    hmap = hazard_map(
        hazard,
        size,
    )

    prediction = model(
        noisy,
        condition,
        hmap,
        timestep,
    )

    reconstructed = project_physics(
        prediction
    )

    loss = physics_loss(
        reconstructed,
        condition,
    )

    print("=" * 100)
    print(
        "VATAVARAN V16 - SMOKE TEST"
    )
    print("=" * 100)

    print(
        "PyTorch           :",
        torch.__version__,
    )

    print(
        "Model output      :",
        tuple(
            prediction.shape
        ),
    )

    print(
        "Physics loss      :",
        float(loss.detach()),
    )

    print(
        "Status            : PASS"
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "train",
            "infer",
        ],
        default="smoke",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--fine-size",
        type=int,
        default=DEFAULT_FINE_SIZE,
    )

    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=DEFAULT_DIFFUSION_STEPS,
    )

    parser.add_argument(
        "--ensemble",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lat",
        type=float,
        default=25.5941,
    )

    parser.add_argument(
        "--lon",
        type=float,
        default=85.1376,
    )

    parser.add_argument(
        "--hazard",
        type=str,
        default="EXTREME_PRECIPITATION",
    )

    args = parser.parse_args()

    if args.mode == "smoke":
        smoke_test()
        return

    if args.mode == "train":
        train_model(
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size,
            fine_size=args.fine_size,
            diffusion_steps=args.diffusion_steps,
            device_name="cpu",
        )
        return

    if args.mode == "infer":

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                "V16 checkpoint not found. "
                "Run --mode train first."
            )

        result = demo_inference(
            center_lat=args.lat,
            center_lon=args.lon,
            hazard=args.hazard,
            ensemble_size=args.ensemble,
            device_name="cpu",
        )

        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key != "features"
                },
                indent=2,
            )
        )

        print(
            "GeoJSON features:",
            len(
                result["features"]
            ),
        )


if __name__ == "__main__":
    main()



