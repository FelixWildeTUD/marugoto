# %%
import os
import json
from typing import Sequence
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import PIL
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import h5py

from . import __version__


__all__ = ["extract_features_"]


class VideoDataset(Dataset):
    def __init__(
        self, video_dir: Path, transform=None,
    ) -> None:
        self.images = list(video_dir.glob("*.jpg"))
        assert self.images, f"no images found in {video_dir}"
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = PIL.Image.open(self.images[i])
        if self.transform:
            image = self.transform(image)

        return image

def extract_features_(
    *,
    model,
    model_name,
    video_paths: Sequence[Path],
    outdir: Path,
) -> None:
    """Extracts features from video images.

    Args:
        video_paths:  A list of paths containing the video images, one per video
        outdir:  Path to save the features to.
    """
    normal_transform = transforms.Compose(
        [
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    extractor_string = f"marugoto-extract_wilde-v{__version__}_{model_name}"
    with open(outdir / "info.json", "w") as f:
        json.dump(
            {
                "extractor": extractor_string,
            },
            f,
        )

    for video_path in tqdm(video_paths):
        video_path = Path(video_path)
        h5outpath = outdir / f"{video_path.name}.h5"
        if (h5outpath).exists():
            print(f"{h5outpath} already exists.  Skipping...")
            continue
        if not next(video_path.glob("*.jpg"), False):
            print(f"No images in {video_path}.  Skipping...")
            continue

        ds = VideoDataset(video_path, normal_transform)

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=64,
            shuffle=False,
            num_workers=os.cpu_count(),
            drop_last=False,
        )

        model = model.eval()

        feats = []
        for batch in tqdm(dl, leave=False):
            feats.append(
                model(batch.type_as(next(model.parameters()))).half().cpu().detach()
            )

        with h5py.File(h5outpath, "w") as f:
            f["feats"] = torch.concat(feats).cpu().numpy()
            f.attrs["extractor"] = extractor_string

if __name__ == "__main__":
    import fire

    fire.Fire(extract_features_)

# %%
