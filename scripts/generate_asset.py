#!/usr/bin/env python3
"""Standalone text -> 3D mesh generator for a single asset.

This is the "fast path" for people who only want a mesh: give it a description
(e.g. a sea creature) and it runs just the first two stages of the full
text-to-3D pipeline:

    1. Text -> Image   (cloud API: OpenAI gpt-image-1.5 OR Google Gemini)
    2. Image -> Mesh   (local SAM3D geometry-generation server)

It deliberately skips everything the full ``AssetManager`` does afterwards
(GLB->GLTF conversion, floater removal, VLM physics analysis, canonicalization,
scaling, convex decomposition, Drake SDF generation). The output is the raw
``.glb`` mesh plus the intermediate ``.png`` image.

Prerequisites:
    * Stage 1 needs a cloud image API key:
        - ``--backend openai``     -> set ``OPENAI_API_KEY``
        - ``--backend gemini``     -> set ``GOOGLE_API_KEY``
        - ``--backend openrouter`` -> set ``OPENROUTER_API_KEY``
      (OpenRouter serves image generation only via its chat-completions
      image models, e.g. google/gemini-2.5-flash-image-preview.)
    * Stage 2 needs the geometry-generation server running with the SAM3D
      backend. See ASSET_GENERATION_QUICKSTART.md for setup, then start it:

        python -m reefsmith.agent_utils.geometry_generation_server.standalone_server \
            --backend sam3d \
            --sam3-checkpoint external/checkpoints/sam3.pt \
            --sam3d-checkpoint external/checkpoints/pipeline.yaml

Example:
    python scripts/generate_asset.py \
        --description "a bright orange clownfish" \
        --backend openai \
        --output-dir ./asset_output
"""

import argparse
import logging
import re
import sys

from pathlib import Path

from omegaconf import OmegaConf

from reefsmith.agent_utils.geometry_generation_server.client import (
    GeometryGenerationClient,
)
from reefsmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationError,
    GeometryGenerationServerRequest,
)
from reefsmith.agent_utils.image_generation import create_image_generator

console_logger = logging.getLogger(__name__)

# Default style prompt fed to the image model. Kept generic so it works for any
# subject; override with --style if you want a specific look.
DEFAULT_STYLE_PROMPT = "photorealistic natural coloring, neutral studio lighting"


def slugify(text: str, max_length: int = 40) -> str:
    """Turn a free-form description into a filesystem-safe short name."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_length] or "asset"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a single 3D mesh (.glb) from a text description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="What to generate, e.g. 'a bright orange clownfish'.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "gemini", "openrouter"],
        default="openai",
        help="Cloud image-generation backend for Stage 1 (default: %(default)s). "
        "'openai' reads OPENAI_API_KEY; 'gemini' reads GOOGLE_API_KEY; "
        "'openrouter' reads OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("asset_output"),
        help="Directory for the generated .png and .glb (default: %(default)s).",
    )
    parser.add_argument(
        "--short-name",
        type=str,
        default=None,
        help="Filesystem-safe base name for outputs. Derived from the "
        "description if not given.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=DEFAULT_STYLE_PROMPT,
        help="Style prompt passed to the image model (default: %(default)r).",
    )

    # Stage 1 (image) tuning.
    parser.add_argument(
        "--image-quality",
        type=str,
        choices=["auto", "low", "medium", "high"],
        default="high",
        help="OpenAI image quality (default: %(default)s). Ignored for gemini.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="1:1",
        help="Gemini aspect ratio (default: %(default)s). Ignored for openai.",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default="1K",
        help="Gemini image size: 1K, 2K, or 4K (default: %(default)s). "
        "Ignored for openai.",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="google/gemini-2.5-flash-image-preview",
        help="OpenRouter image model slug (default: %(default)s). Used only when "
        "--backend openrouter.",
    )

    # Stage 2 (geometry server).
    parser.add_argument(
        "--geometry-host",
        type=str,
        default="127.0.0.1",
        help="Geometry-generation server host (default: %(default)s).",
    )
    parser.add_argument(
        "--geometry-port",
        type=int,
        default=7000,
        help="Geometry-generation server port (default: %(default)s).",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=str,
        default="external/checkpoints/sam3.pt",
        help="Path to the SAM3 segmentation checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--sam3d-checkpoint",
        type=str,
        default="external/checkpoints/pipeline.yaml",
        help="Path to the SAM 3D Objects pipeline config (default: %(default)s).",
    )
    parser.add_argument(
        "--sam3d-mode",
        type=str,
        choices=["foreground", "object_description"],
        default="foreground",
        help="SAM3D segmentation mode (default: %(default)s). 'foreground' "
        "auto-segments the subject off the plain studio background (best for "
        "these images); 'object_description' segments using the description text.",
    )
    parser.add_argument(
        "--sam3d-threshold",
        type=float,
        default=0.5,
        help="SAM3D mask confidence threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args()


def build_image_generator(args: argparse.Namespace):
    """Create the Stage 1 image generator with a minimal inline config."""
    config = OmegaConf.create(
        {
            "openai": {"quality": args.image_quality},
            "gemini": {
                "aspect_ratio": args.aspect_ratio,
                "image_size": args.image_size,
            },
            "openrouter": {"model": args.openrouter_model},
        }
    )
    return create_image_generator(backend=args.backend, config=config)


def generate_image(args: argparse.Namespace, image_path: Path) -> None:
    """Stage 1: text -> image."""
    console_logger.info(
        "Stage 1/2: generating image via %s -> %s", args.backend, image_path
    )
    generator = build_image_generator(args)
    generator.generate_images(
        style_prompt=args.style,
        object_descriptions=[args.description],
        output_paths=[image_path],
        labels=[args.short_name],
    )
    if not image_path.exists():
        raise RuntimeError(f"Image generation did not produce {image_path}")
    console_logger.info("Stage 1/2 complete: %s", image_path)


def generate_mesh(
    args: argparse.Namespace, image_path: Path, output_dir: Path
) -> str:
    """Stage 2: image -> mesh (.glb) via the SAM3D geometry server."""
    client = GeometryGenerationClient(
        host=args.geometry_host, port=args.geometry_port
    )

    if not client.health_check():
        raise ConnectionError(
            f"Geometry-generation server is not reachable at "
            f"{args.geometry_host}:{args.geometry_port}.\n"
            "Start it first (see ASSET_GENERATION_QUICKSTART.md):\n"
            "  python -m reefsmith.agent_utils.geometry_generation_server."
            "standalone_server \\\n"
            "      --backend sam3d \\\n"
            f"      --sam3-checkpoint {args.sam3_checkpoint} \\\n"
            f"      --sam3d-checkpoint {args.sam3d_checkpoint}"
        )

    sam3d_config = {
        "sam3_checkpoint": args.sam3_checkpoint,
        "sam3d_checkpoint": args.sam3d_checkpoint,
        "mode": args.sam3d_mode,
        "object_description": (
            args.description if args.sam3d_mode == "object_description" else None
        ),
        "threshold": args.sam3d_threshold,
    }

    request = GeometryGenerationServerRequest(
        image_path=str(image_path.resolve()),
        output_dir=str(output_dir.resolve()),
        prompt=args.description,
        output_filename=f"{args.short_name}.glb",
        backend="sam3d",
        sam3d_config=sam3d_config,
    )

    console_logger.info("Stage 2/2: generating mesh via SAM3D (this can take minutes)")
    geometry_path: str | None = None
    for _index, result in client.generate_geometries([request]):
        if isinstance(result, GeometryGenerationError):
            raise RuntimeError(
                f"Mesh generation failed: {result.error_message}"
            )
        geometry_path = result.geometry_path

    if geometry_path is None:
        raise RuntimeError("Geometry server returned no result.")

    console_logger.info("Stage 2/2 complete: %s", geometry_path)
    return geometry_path


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.short_name is None:
        args.short_name = slugify(args.description)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.short_name}.png"

    try:
        generate_image(args, image_path)
        geometry_path = generate_mesh(args, image_path, output_dir)
    except (ConnectionError, RuntimeError, ValueError) as e:
        console_logger.error("%s", e)
        return 1

    print()
    print("Done.")
    print(f"  Image: {image_path}")
    print(f"  Mesh:  {geometry_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
