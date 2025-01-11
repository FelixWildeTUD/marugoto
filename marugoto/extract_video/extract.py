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
        self, video_dir: Path, transform=None, *, repetitions: int = 1
    ) -> None:
        self.images = list(video_dir.glob("*.jpg"))
        assert self.images, f"no images found in {video_dir}"
        self.images *= repetitions
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
    augmented: bool = False,
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """
    normal_transform = transforms.Compose(
        [
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    augmenting_transform = transforms.Compose(
        [
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.2, saturation=0.25, hue=0.125
                    )
                ],
                p=0.5,
            ),
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
                "augmented": augmented,
            },
            f,
        )

    for video_path in tqdm(video_paths):
        video_path = Path(video_path)
        if not augmented:
            h5outpath = outdir / f"{video_path.name}.h5"
        else:
            h5outpath = outdir / f"{video_path.name}_augmented.h5"
        if (h5outpath).exists():
            print(f"{h5outpath} already exists.  Skipping...")
            continue
        if not next(video_path.glob("*.jpg"), False):
            print(f"No images in {video_path}.  Skipping...")
            continue
        
        
        '''unaugmented_ds = VideoDataset(video_path, normal_transform)
        augmented_ds = VideoDataset(
            video_path, augmenting_transform, repetitions=augmented_repetitions
        )'''

        if not augmented:
            ds = VideoDataset(video_path, normal_transform)
        else:
            ds = VideoDataset(video_path, augmenting_transform)

        #ds = ConcatDataset([unaugmented_ds, augmented_ds])
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
            '''f["augmented"] = np.repeat(
                [False, True], [len(unaugmented_ds), len(augmented_ds)]
            )'''
            #assert len(f["feats"]) == len(f["augmented"])
            f.attrs["extractor"] = extractor_string


if __name__ == "__main__":
    import fire

    fire.Fire(extract_features_)

# %%
