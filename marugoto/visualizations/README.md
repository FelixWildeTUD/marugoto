# Heatmap script

This script is designed to process feature files and generate heatmaps and extracting the top tiles using an exported learner using the ViT regression. The script takes five input arguments:

- `--learner_path`: Path to the exported learner (.pkl) file.
- `--feature_name_pattern`: Pattern to match feature files (e.g., '/path/to/files/*.h5').
- `--output_folder`: Path to the output folder where results will be saved.
- `--wsi_dir`: Path to the folder containing the WSIs.
- `--n_toptiles `: Number of toptiles to generate, 8 by default.


### Example

```bash
python transformer_heatmap.py \
    --learner_path "/path/to/export.pkl" \
    --feature_name_pattern "/path/to/slide/features/*.h5" \
    --output_folder "path/to/store/output"
    --wsi_dir "path/to/wsi/images"
    --n_toptiles 8
```

### Output

```bash
      Results_folder
      ├── Input_slide_name
      │   ├── toptiles
      │   │   ├── toptiles_1_(x,y).jpg
      │   │   ├── toptiles_2_(x,y).jpg
      │   │   ├── ...
      │   │   └── toptiles_8_(x,y).jpg
      │   │
      │   ├── slide_thumbnail.jpg
      │   ├── Input_slide_name_attention_map_layer_0.png
      │   └── Input_slide_name_toptiles_layer_0.csv
      ├── Input_slide_name_1
      └── ...
   ```
