#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import random

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response

def format_data(lines):
    """from lines in dataset to two lists of sentences and labels respectively"""
    def process_raw_data_sst(lines):
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    train_sentences, train_labels = process_raw_data_sst(lines)
    return train_sentences, train_labels


class SST2Decomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = [0, 1]
        num_per_class = int(np.ceil(k_shot / len(labels)))
        print(f"Selecting {num_per_class} examples per class.")

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1
            sub_df = train_data[train_data["label"] == label].sample(num_per_class)
            dfs.append(sub_df)
            total_in_context += num_per_class
            if total_in_context == k_shot:
                break
        mini_df = pd.concat(dfs)
        return mini_df

    def read_data(self, save_dir, overwrite_data):
        save_data = Path(f"{save_dir}/{self.task_name}/data.feather")
        if not save_data.exists() or overwrite_data:
            with open(f"{self.data_dir}/stsa.binary.test", "r") as f:
                test_lines = f.readlines()
            test_sentences, test_labels = format_data(test_lines)
            test_data = pd.DataFrame({
                'sentence': test_sentences,
                'label': test_labels,
            })
            test_data.to_feather(f"{save_data}")
        else:
            print(f"Reading test data from {save_data}")
            test_data = pd.read_feather(save_data)

        save_data = Path(f"{save_dir}/{self.task_name}/train_data.feather")
        if not save_data.exists() or overwrite_data:
            with open(f"{self.data_dir}/stsa.binary.train", "r") as f:
                train_lines = f.readlines()
            train_sentences, train_labels = format_data(train_lines)
            train_data = pd.DataFrame({
                'sentence': train_sentences,
                'label': train_labels,
            })
            train_data.to_feather(f"{save_data}")
        else:
            print(f"Reading train data from {save_data}")
            train_data = pd.read_feather(save_data)
        print(f"Test Data Size: {len(test_data)}")
        print(f"Train Data Size: {len(train_data)}")
        return test_data, train_data


    def get_boost_decomp_examples(self, train_data, boost_id):
        seed = [1, 2, 3][boost_id]
        k_shot = 16
        random.seed(seed)
        np.random.seed(seed)

        data_train = pd.DataFrame(train_data)
        labels = set(data_train["label"])
        num_per_class = int(np.ceil(k_shot / len(labels)))

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1

            if seed % 2 == 1:
                sub_df = data_train[data_train["label"] == label].sample(num_per_class, random_state=seed)
            elif seed % 2 == 0:
                sub_df = data_train[data_train["label"] != label].sample(num_per_class, random_state=seed)
            dfs.append(sub_df)
            total_in_context += num_per_class
            if total_in_context == k_shot:
                break

        booster_df = pd.concat(dfs).sample(frac=1, random_state=0)
        print(f"Selected: {len(booster_df)} in context examples.")
        return [
            booster_df
        ]

    def zero_few_baseline(
        self,
        test_data,
        few_shot_df,
        manifest,
        overwrite_manifest,
        do_few_shot=True,
    ):
        expt_log = {}
        preds = []
        labels = []

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            if ind in expt_log:
                pred = entry["pred"]
                gold_str = entry["gold"]
            else:
                sentence = row["sentence"]
                label = row["label"]

                icl_str = ""
                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        if s_row["label"] == 0:
                            demo_label = "negative"
                        else:
                            demo_label = "positive"
                        icl = f"Text: {s_row['sentence']}\nSentiment: {demo_label}"
                        icl_str += f"{icl}\n\n"

                description = "For each snippet of text, label the sentiment of the text as positive or negative."
                prompt = f"{description}\n\n{icl_str}Text: {{sentence:}}\nSentiment:"
                pmp = prompt.format(sentence=sentence)
                if i == 0:
                    print(pmp)

                pred = get_response(
                    pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=50,
                )
                pred = (
                    pred.replace(".", "")
                    .replace(",", "")
                    .replace("Label: ", "")
                    .replace("Sentiment: ", "")
                )
                pred = [p for p in pred.split("\n") if p]
                is_pos = "positive" in pred
                is_neg = "negative" in pred
                if is_pos and not is_neg:
                    pred = "positive"
                elif is_neg and not is_pos:
                    pred = "negative"
                else:
                    pred = ""

                if label == 1:
                    gold_str = "positive"
                else:
                    gold_str = "negative"

                entry = {
                    "gold": gold_str,
                    "pred": pred,
                    "base_prompt": pmp,
                    "ind": ind,
                    "example": sentence,
                }
                expt_log[ind] = entry

            preds.append(pred)
            labels.append(gold_str)

        report = classification_report(labels, preds, output_dict=True)
        return expt_log, report["accuracy"]

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest, run_limit=-1)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest, run_limit=1000)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train)
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            individual_accuracies.append(classification_report(labels, [p[i] for p in all_boost_preds], output_dict=True)["accuracy"])
        report = classification_report(labels, preds, output_dict=True)
        return expt_log, expt_log_train, report["accuracy"], individual_accuracies

    def _run_decomp_single_data(self, test_data, boost_dfs, manifest, overwrite_manifest, run_limit=-1):
        expt_log = {}
        all_boost_preds = []
        labels = []
        
        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            sentence = row["sentence"]
            label = row["label"]
            
            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                all_prompts = []
                icl_str = ""
                for s_ind, s_row in boost_examples[0].iterrows():
                    if s_row["label"] == 0:
                        demo_label = "negative"
                    else:
                        demo_label = "positive"
                    icl = f"Text: {s_row['sentence']}\nSentiment: {demo_label}"
                    icl_str += f"{icl}\n\n"

                description = "For each snippet of text, label the sentiment of the text as positive or negative."
                prompt = f"{description}\n\n{icl_str}Text: {{sentence:}}\nSentiment:"
                pmp = prompt.format(sentence=sentence)
                all_prompts.append(pmp)
                if i == 0:
                    print(pmp)
                pred = get_response(
                    pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=5,
                )
                pred = pred.replace(".", "").replace(",", "").replace("Label: ", "")
                pred = [p for p in pred.split("\n") if p]
                if pred:
                    pred = pred[0]
                else:
                    pred = ""
                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)

            if label == 1:
                gold_str = "positive"
            else:
                gold_str = "negative"

            entry = {
                "gold": gold_str,
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "example": sentence,
                "ind": i,
            }
            expt_log[i] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(gold_str)

        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 3
    task_name = "sst2"
    data_dir = f"{DATA_DIR}/sst2/"
    if not Path(data_dir).exists():
        raise ValueError(
            f"Data directory {data_dir} does not exist. Download AGNews from https://github.com/tonyzhaozh/few-shot-learning.")
    decomp = SST2Decomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
