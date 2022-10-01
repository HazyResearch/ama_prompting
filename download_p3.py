import datasets
import tensorflow as tf
import pandas as pd
from pathlib import Path
import json
from tqdm.auto import tqdm
import os

DATA_DIR = os.environ.get("AMA_DATA", "/home/data")

# Download P3 github data from HF website
'''
git lfs install
git clone https://huggingface.co/datasets/bigscience/P3
'''

import sys
sys.path.append(f"{DATA_DIR}/P3")
# From P3 github
from tasks_splits_and_features import DATA_SPLITS_SIZES

SPLITS = ["train", "test", "validation"]
_FEAT_MAPPING_FUNCTIONS = {
    "answer_choices": lambda x: [choice.decode("utf-8") for choice in x],
    "inputs": lambda x: x.tolist(),
    "inputs_pretokenized": lambda x: x.decode("utf-8"),
    "targets": lambda x: x.tolist(),
    "targets_pretokenized": lambda x: x.decode("utf-8"),
    "idx": lambda x: x.tolist(),
    "weight": lambda x: float(x),
    "is_correct": lambda x: x,
}
def _feature_config(shape, dtype):
    if dtype in ("int32", "bool"):
        # int32 and bool are stored as int64 in the tf.train.Example protobuf.
        dtype = "int64"
    if shape and shape[0] is None:
        return tf.io.FixedLenSequenceFeature(
            shape[1:], dtype, allow_missing=True
        )
    return tf.io.FixedLenFeature(shape, dtype)

@tf.autograph.experimental.do_not_convert
def extract_dataset(data_path, subfolder, sizes, processes=10, splits=SPLITS):
    datasets = {}
    for split in splits:
        if not (data_path / subfolder / f"info.{split}.json").exists():
            continue
        features_dict = json.load(open(data_path / subfolder / f"info.{split}.json"))
        if "features" not in features_dict:
            features_dict = json.load(open(data_path / subfolder / f"info.train.json"))
        if "features" not in features_dict:
            continue
        features_dict = features_dict["features"]
        tfrecord = str(data_path / subfolder / f"{split}.tfrecord-00000-of-00001")
        feature_description = {
            feat: _feature_config(**desc) for feat, desc in features_dict.items()
        }
        ds = tf.data.TFRecordDataset(tf.io.gfile.glob([tfrecord])) # TODO -> handle multiple shards
        ds = ds.map(
            lambda pb: tf.io.parse_single_example(pb, feature_description),
            num_parallel_calls=processes
        )
        # Cast features back to the types from the info JSON since some features
        # must be cast for storage (e.g., int32 is stored as int64).
        ds = ds.map(
            lambda x: {k: tf.cast(v, features_dict[k]["dtype"]) for k, v in x.items()},
            num_parallel_calls=processes
        )
        res = []
        for ex in tqdm(ds.as_numpy_iterator(), total=sizes.get(split, 10000)):
            ex_dict = {}
            for feat_name, feat_value in ex.items():
                ex_dict[feat_name] = _FEAT_MAPPING_FUNCTIONS[feat_name](feat_value)
            res.append(ex_dict)
        if len(res) > 0:
            df = pd.DataFrame.from_records(res)
            datasets[split] = df
    return datasets

data_path = Path(f"{DATA_DIR}/P3/data")
output_data_path = Path(f"{DATA_DIR}/P3/data_feather")
for subfolder in data_path.iterdir():
    print(subfolder)
    subfolder = subfolder.name
    out_path = output_data_path / subfolder
    out_path.mkdir(parents=True, exist_ok=True)
    splits_to_pass = []
    for split in SPLITS:
        if not (out_path / f"{split}.feather").exists():
            splits_to_pass.append(split)
    datasets = extract_dataset(data_path, subfolder, sizes=DATA_SPLITS_SIZES[str(subfolder)], splits=splits_to_pass)
    for split, df in datasets.items():
        df.to_feather(out_path / f"{split}.feather")