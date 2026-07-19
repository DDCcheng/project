# data/

The dataset images are **local-only and must not be committed to git**. Each
team member downloads or extracts the Kaggle dataset separately and passes its
local path to the preprocessing scripts.

## Layout

One folder per class (20 classes: 5 fruits + 5 vegetables, each fresh/rotten).
The scripts infer the class label from the immediate parent folder of each image.

```
data/<dataset-folder>/
    freshapples/    img001.jpg ...
    rottenapples/   ...
    ...             (20 class folders total)
```

Point the preprocessing scripts at this folder with
`--data-dir data/<dataset-folder>`.

## Downloading or re-downloading

Source: <https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset>
or run `python src/preprocessing/download_data.py` (needs Kaggle credentials).
