# Quickstart: text → 3D mesh (assets only)

This is the **fast path** for the common case: *"give me a description of a sea
creature (or any object) and hand back a 3D mesh."* It runs only the first two
stages of the full text-to-3D pipeline and skips all the simulation-prep work
(physics analysis, collision decomposition, Drake SDF, etc.).

```
"a bright orange clownfish"
        │
   Stage 1: Text → Image      ☁️  cloud API (OpenAI or Gemini)
        │   clownfish.png
        ▼
   Stage 2: Image → Mesh      🖥️  local GPU (SAM3 + SAM3D server)
        │
   clownfish.glb   ← the mesh
```

Driver script: [`scripts/generate_asset.py`](scripts/generate_asset.py).

---

## About API keys (read this first)

There are **two** distinct systems, not one:

| Stage | Runs where | Needs |
|-------|-----------|-------|
| 1. Text → Image | Cloud API | An **OpenAI**, **Google/Gemini**, or **OpenRouter** key |
| 2. Image → Mesh | **Your GPU** | No key — just checkpoints + a running server |

Stage 1 has three backends (pick with `--backend`):

| `--backend` | Key env var | Notes |
|-------------|-------------|-------|
| `openai` (default) | `OPENAI_API_KEY` | `gpt-image-1.5` via the Images API |
| `gemini` | `GOOGLE_API_KEY` | `gemini-3-pro-image-preview` |
| `openrouter` | `OPENROUTER_API_KEY` | Via OpenRouter's chat-completions **image** models |

> ℹ️ **About OpenRouter:** it does *not* implement the OpenAI Images API, but it
> can generate images through its chat-completions endpoint using image-output
> models (default `google/gemini-2.5-flash-image-preview`, override with
> `--openrouter-model`). Caveats: the set of image-capable models on OpenRouter
> changes over time, and this backend supports asset image generation only (no
> image-edit / context images).

---

## One-time setup

### 1. Install the Python environment

```bash
uv sync
```

### 2. Install SAM3D + SAM3 (Stage 2)

Requires a CUDA 12.x GPU with **≥ 32 GB** memory.

```bash
# Request access on HuggingFace first (approval can take a bit):
#   https://huggingface.co/facebook/sam3
#   https://huggingface.co/facebook/sam-3d-objects
hf auth login          # or: huggingface-cli login

bash scripts/install_sam3d.sh
```

This clones the SAM3 and SAM 3D Objects repos into `external/`, builds the CUDA
dependencies (pytorch3d, gsplat, nvdiffrast, kaolin, …), and downloads the
checkpoints to `external/checkpoints/`:

- `external/checkpoints/sam3.pt`         — SAM3 segmentation model
- `external/checkpoints/pipeline.yaml`   — SAM 3D Objects pipeline config

Budget ~20–40 minutes for this the first time.

### 3. Set your image-API key

```bash
# Pick ONE, matching the --backend you'll use:
export OPENAI_API_KEY="sk-..."          # for --backend openai (default)
export GOOGLE_API_KEY="..."             # for --backend gemini
export OPENROUTER_API_KEY="sk-or-..."   # for --backend openrouter
```

---

## Running

### Step A — start the geometry server (leave it running)

```bash
python -m reefsmith.agent_utils.geometry_generation_server.standalone_server \
    --backend sam3d \
    --sam3-checkpoint external/checkpoints/sam3.pt \
    --sam3d-checkpoint external/checkpoints/pipeline.yaml
```

Loads the models onto the GPU and listens on `127.0.0.1:7000`. Start it once and
reuse it for many assets.

### Step B — generate an asset

In another shell:

```bash
python scripts/generate_asset.py \
    --description "a bright orange clownfish" \
    --backend openai \
    --output-dir ./asset_output
```

Output:

```
asset_output/
├── a_bright_orange_clownfish.png    # Stage 1 image
└── a_bright_orange_clownfish.glb    # Stage 2 mesh  ← this is what you want
```

Switch the image backend with `--backend gemini` (reads `GOOGLE_API_KEY`) or
`--backend openrouter` (reads `OPENROUTER_API_KEY`), e.g.:

```bash
python scripts/generate_asset.py \
    --description "a bright orange clownfish" \
    --backend openrouter \
    --openrouter-model google/gemini-2.5-flash-image-preview \
    --output-dir ./asset_output
```

---

## Useful flags

| Flag | Purpose | Default |
|------|---------|---------|
| `--description` | What to generate (**required**) | — |
| `--backend` | `openai`, `gemini`, or `openrouter` for Stage 1 | `openai` |
| `--openrouter-model` | OpenRouter image model slug | `google/gemini-2.5-flash-image-preview` |
| `--output-dir` | Where to write `.png` + `.glb` | `asset_output` |
| `--short-name` | Base filename for outputs | derived from description |
| `--style` | Style prompt for the image model | neutral studio look |
| `--image-quality` | OpenAI quality: auto/low/medium/high | `high` |
| `--aspect-ratio` / `--image-size` | Gemini image controls | `1:1` / `1K` |
| `--geometry-host` / `--geometry-port` | Where the server is listening | `127.0.0.1` / `7000` |
| `--sam3d-mode` | `foreground` (auto) or `object_description` | `foreground` |

Run `python scripts/generate_asset.py --help` for the full list.

---

## How the prompt is built (Stage 1)

The image prompt comes from
[`reefsmith/prompts/data/image_generation/asset_image_initial.yaml`](reefsmith/prompts/data/image_generation/asset_image_initial.yaml),
filled with your `--description` and `--style`. It instructs the model to render
the subject as a single, isolated object on a plain, opaque, contrasting
background — no water tint, caustics, particles, or shadows — which is exactly
what SAM3D needs to segment a clean mesh in Stage 2.

## Troubleshooting

- **"Geometry-generation server is not reachable"** — Step A isn't running, or
  you used a different `--geometry-port`. Start the server (or point the script
  at the right host/port).
- **`OPENAI_API_KEY environment variable is required`** — export the key for the
  backend you chose (or switch `--backend`).
- **Mesh generation failed / segmentation empty** — try
  `--sam3d-mode object_description`, or regenerate the image (a cleaner,
  well-isolated subject on a plain background segments best).
