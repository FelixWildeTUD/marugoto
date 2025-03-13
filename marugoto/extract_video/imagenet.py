#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Resnet18 imagenet features from video."
    )
    parser.add_argument(
        "video_paths",
        metavar="VIDEO_DIR",
        type=str,
        help="A directory with images from a video.",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, required=True, help="Path to save the features to."
    )
    args = parser.parse_args()
    print(f"{args=}")

import torch
from .extract import extract_features_

__all__ = ["extract_resnet18_imagenet_features"]


def extract_resnet18_imagenet_features_(video_paths, **kwargs):
    """Extracts features from video images.

    Args:
        video_paths:  A list of paths containing the video images, one
            per video.
        outdir:  Path to save the features to.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)

    video_paths = glob.glob(video_paths)

    return extract_features_(
        video_paths=video_paths,
        model=model,
        model_name="resnet18-imagenet",
        **kwargs,
    )

if __name__ == "__main__":
    extract_resnet18_imagenet_features_(**vars(args))
