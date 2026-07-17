# data/

The dataset images live here and **are committed to git**, so you get them
directly from `git pull` — no Kaggle account or download needed.

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

## Re-downloading (only if you need a fresh copy)

Source: <https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset>
or run `python src/preprocessing/download_data.py` (needs Kaggle credentials).
