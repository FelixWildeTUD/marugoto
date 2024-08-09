#!/usr/bin/env python3

__author__ = "Omar S. M. El Nahhas, Sandro Carollo"
__copyright__ = "Copyright 2024, Kather Lab"
__license__ = "MIT"
__maintainer__ = ["Omar S. M. El Nahhas"]
__email__ = "omar.el_nahhas@tu-dresden.de"

# %%
from collections import namedtuple
from functools import partial

import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from einops import rearrange
from fastai.vision.all import load_learner
import pandas as pd
import os
import re
import glob
import argparse
import openslide
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

# %%

# Helper function for top tile coordinates
def get_toptile_coords(scores, coords):
    coords_list = np.array([str(tuple(foo)) for foo in coords])
    return pd.DataFrame({'gradcam': scores, 'coords': coords_list})

def toptile_coords(list_of_df):
    full_df = pd.concat(list_of_df)
    mean_scores_coords = full_df.groupby('coords').mean('gradcam').reset_index()
    top_n_tile_coords = mean_scores_coords.sort_values(by='gradcam', ascending=False)
    return top_n_tile_coords

def vals_to_im(scores, coords, stride):
    size = coords.max(0)[::-1]
    if scores.ndimension() == 1:
        im = np.zeros(size)
    elif scores.ndimension() == 2:
        im = np.zeros((*size, scores.size(-1)))
    else:
        raise ValueError(f"{scores.ndimension()=}")
    for score, c in zip(scores, coords[:]):
        x, y = c[0], c[1]
        im[y:(y+stride), x:(x+stride)] = score.cpu().detach().numpy()
    return im

def save_qkv(module, input, output):
    global q, k, v

    # Capture the QKV matrices
    qkv = output.chunk(3, dim=-1)
    n_heads = 8  # Model has 8 attention heads

    # Rearrange the dimensions for processing
    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=n_heads), qkv)

def extract_positions(coords):
    # Use regex to find numbers in the string
    match = re.findall(r'\d+', coords)
    if match:
        pos_0 = int(match[0])
        pos_1 = int(match[1])
        return pos_0, pos_1
    return None, None

def get_n_toptiles(
    slide,
    output_dir,
    scores,
    stride: int,
    n: int = 8,
    tile_size: int = 512,
    thumbnail_size: tuple = (2048, 2048)
) -> None:
    slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])

    # determine the scaling factor between heatmap and original slide
    # 256 microns edge length by default, with 224px = ~1.14 MPP (± 10x magnification)
    feature_downsample_mpp = (
        256 / stride
    )  # NOTE: stride here only makes sense if the tiles were NON-OVERLAPPING
    scaling_factor = feature_downsample_mpp / slide_mpp

    top_score = scores.head(n).reset_index()

    # OPTIONAL: if the score is not larger than 0.5, it's indecisive on directionality
    # then add [top_score.values > 0.5]
    for index, row in top_score.iterrows():
        # Extract positions from the score row
        pos_0, pos_1 = extract_positions(row['coords'])
        
        # Ensure positions are valid
        if pos_0 is None or pos_1 is None:
            print(f"[ERROR] Invalid coordinates: {row['coords']}")
            continue

        # Scale positions to match slide resolution
        scaled_pos_0 = int(pos_0 * scaling_factor)
        scaled_pos_1 = int(pos_1 * scaling_factor)

        # Define target tile size
        target_size = (int(2 * stride * scaling_factor), int(2 * stride * scaling_factor))

        # Read the region from the slide
        tile = (
            slide.read_region(
                (scaled_pos_0, scaled_pos_1),
                0,
                target_size
            )
            .convert("RGB")
            .resize((tile_size, tile_size))  # Resize to desired tile size
        )

        # Construct filename and save path
        tile_filename = f"toptiles_{index+1}_({pos_0},{pos_1}).jpg"
        tile_output_dir = output_dir / "toptiles"
        tile_output_dir.mkdir(exist_ok=True, parents=True)
        tile_path = tile_output_dir / tile_filename

        # Save the tile
        try:
            tile.save(tile_path)
            print(f"[TOP TILES] Tile saved at {tile_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save tile at {tile_path}: {e}")
        
    # Create and save a thumbnail of the entire slide
    thumbnail = slide.get_thumbnail(thumbnail_size)  # Resize the entire slide to desired size
    thumbnail_filename = "slide_thumbnail.jpg"
    thumbnail_path = output_dir / thumbnail_filename
    thumbnail.save(thumbnail_path)
    print(f"Thumbnail saved at {thumbnail_path}")

def main(learner_path, feature_name_pattern, output_folder, wsi_dir, n_toptiles):
    # Load the learner
    learn = load_learner(learner_path)

    # Find all matching feature files
    feature_files = glob.glob(feature_name_pattern)
    learn.model.cuda().eval()

    # Supported OpenSlide extensions
    supported_extensions = ['.ndpi', '.svs', '.tiff', '.tif', '.vms', '.vmu', 
                            '.scn', '.mrxs', '.tiff', '.bif', '.jpg', '.png']
    
    # Traverse each feature file for processing
    for file in feature_files:
        f_hname = os.path.basename(file) 
        base_name = os.path.splitext(f_hname)[0]     
        output_folder = Path(output_folder)
        output_folder = output_folder / base_name
        output_folder.mkdir(exist_ok=True, parents=True)
        print(f'Processing {f_hname}...')

        # Check if output file already exists
        if (output_folder / f"{f_hname}_toptiles_layer_0.csv").exists():
            print("Exists. Skipping...")
            continue

        # Load features and coordinates from the HDF5 file
        with h5py.File(file, 'r') as f:
            coords = f["coords"][:]
            feats = torch.tensor(f["feats"][:]).cuda().float()
            # Ensure that coordinates and features are correctly loaded
            print(f"[INFO] H5 coords shape: {coords.shape}")
            print(f"[INFO] H5 feats shape: {feats.shape}")
        # Determine stride from coordinates
        xs = np.sort(np.unique(coords[:, 0]))
        stride = np.min(xs[1:] - xs[:-1])

        # Find the WSI file by checking all supported openslide extensions
        wsi_path = None
        for ext in supported_extensions:
            wsi_filename = base_name + ext
            possible_path = Path(wsi_dir) / wsi_filename

            if possible_path.exists():
                wsi_path = possible_path
                break

        if wsi_path is None:
            print(f"[ERROR] WSI file for {base_name} not found. Skipping...")
            continue

        # Open the WSI file
        slide = openslide.open_slide(wsi_path)

        # Initialize q and k to None for each file
        global q, k, v
        q, k, v = None, None, None

        # Process each layer and attention head
        for transformer_layer in range(1):  # Limiting to first layer
            img_coll = []
            coords_coll = []

            # Iterate over each attention head
            for attention_head_i in range(8):  # 8 attention heads in the model

                # Prepare input features
                feats.requires_grad = True
                embedded = learn.fc(feats.unsqueeze(0).float())
                with_class_token = torch.cat([learn.cls_token, embedded], dim=1)

                # Register forward hook for qkv extraction
                handle = learn.model.transformer.layers[transformer_layer][0].fn.to_qkv.register_forward_hook(save_qkv)

                # Forward pass
                try:
                    transformed = learn.model.transformer(with_class_token)[:, 0]
                except Exception as e:
                    print(f"[ERROR] Exception during forward pass: {e}")
                    continue

                # Remove the hook
                handle.remove()

                # Debugging: Check if q and k are captured correctly
                if q is None or k is None:
                    print(f"[ERROR] Failed to capture qkv matrices. Skipping this head.")
                    continue

                # Calculate attention scores
                try:
                    # Extract the specific head to be processed
                    q_head = q[:, attention_head_i]
                    k_head = k[:, attention_head_i]

                    # Compute scaled dot-product attention
                    a = F.softmax(q_head @ k_head.transpose(-2, -1) * (1.0 / np.sqrt(q.size(-1))), dim=-1)

                    # Verify attention scores (add if condition to check validity)
                    if torch.isnan(a).any() or torch.isinf(a).any():
                        print(f"[ERROR] Invalid values in attention scores. Skipping this head.")
                        continue
                except Exception as e:
                    print(f"[ERROR] Error calculating attention scores: {e}")
                    continue

                # Calculate attention gradcam
                try:
                    # Ensure the backward pass is called correctly for the attention map
                    a[0, 0, 1:].sum().backward(retain_graph=True)  # Use retain_graph if needed for debugging
                    gradcam = (feats.grad * feats).abs().sum(-1)
                    print(f"[INFO] Grad-CAM calculated for head {attention_head_i}. Shape: {gradcam.shape}")

                    X = vals_to_im(gradcam, coords, stride)

                    if transformer_layer == 0:
                        coords_coll.append(get_toptile_coords(gradcam.cpu().detach().numpy(), coords))

                    # Scale and store image
                    if (X.max() - X.min()) > 0:
                        X_std = (X - X.min()) / (X.max() - X.min())
                    else:
                        continue
                    X_scaled = X_std * 255
                    img_coll.append(X_scaled)  # for the aggregated heatmap later, like Firas' paper
                except Exception as e:
                    print(f"[ERROR] Error calculating gradcam: {e}")
                    continue

            # Aggregate heatmaps across heads
            if img_coll:
                img = np.mean(img_coll, axis=0)
                print(f"[INFO] Finished processing {f_hname} for transformer layer {transformer_layer}.")

                if transformer_layer == 0:
                    # Save top tile coordinates
                    if coords_coll:
                        df = toptile_coords(coords_coll)
                        df.to_csv(output_folder / f"{f_hname}_toptiles_layer_{transformer_layer}.csv")
                    else:
                        print(f"[WARNING] No valid top tile coordinates captured for {f_hname}.")

                # Plot and save heatmap with color bar
                fig, ax = plt.subplots()
                
                # Define a custom colormap with white background
                cmap = plt.get_cmap('plasma')
                new_colors = cmap(np.linspace(0, 1, 256))
                new_colors[0] = np.array([1, 1, 1, 1])  # Set the lowest value to white (RGBA)
                white_cmap = LinearSegmentedColormap.from_list('white_plasma', new_colors)
                
                # Enhanced color representation with custom colormap and PowerNorm
                cax = ax.imshow(img, cmap=white_cmap, alpha=0.85, interpolation='nearest', norm=PowerNorm(gamma=0.5))
                cbar = plt.colorbar(cax, ax=ax, orientation='vertical')
                cbar.set_label('Grad-CAM Score')
                ax.axis('off')
                plt.savefig(output_folder / f"{f_hname}_attention_map_layer_{transformer_layer}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free memory
            else:
                print(f"[WARNING] No valid images to process for {f_hname}. Skipping heatmap generation.")
        
        # Top tiles generation part:
        print(f"[TOP TILES] Generation of {n_toptiles} top tiles.")
        print(f"Creating top tiles for {f_hname}...")
        get_n_toptiles(
                slide=slide,
                stride=stride,
                output_dir=output_folder,
                scores=df,
                n=n_toptiles,
                tile_size=512,
                thumbnail_size=(2048, 2048)
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Oncotype project data.")

    parser.add_argument("--learner_path", type=str, required=True,
                        help="Path to the exported learner (.pkl) file.")
    parser.add_argument("--feature_name_pattern", type=str, required=True,
                        help="Pattern to match feature files (e.g., '/path/to/files/*.h5').")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder where results will be saved.")
    parser.add_argument("--wsi_dir", type=Path, required=True,
                        help="Directory path containing the SVSs images")
    parser.add_argument("--n_toptiles", type=int, default=8, required=False,
                        help="Number of toptiles to generate, 8 by default")
    args = parser.parse_args()

    main(args.learner_path, args.feature_name_pattern, args.output_folder, args.wsi_dir, args.n_toptiles)
