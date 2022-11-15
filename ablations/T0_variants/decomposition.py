#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import argparse
from typing import Counter
import pandas as pd
import json
import numpy as np
import datetime
import os
import random

from utils import save_log, get_manifest_session

DATA_DIR = os.environ.get("AMA_DATA", "/home/data")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--num_run", type=int, default=-1, help="Number of rows of test data to run"
    )
    parser.add_argument(
        "--k_shot", type=int, default=3, help="Number of few shot"
    )
    parser.add_argument(
        "--num_boost", type=int, default=3, help="Number of few shot sets to boost over")
    parser.add_argument(
        "--boost_train_examples", type=int, default=1000, help="Number of training examples to run through for boosting"
    )
    parser.add_argument(
        "--output_metrics_file", type=str, default="decomposition_metrics.json", help="Output file for all metrics."
    )
    parser.add_argument(
        "--save_dir", type=str, default="/home/final_runs/", help="Data directory"
    )
    parser.add_argument(
        "--run_decomp",
        type=int,
        default=1,
        help="Run decomp",
        choices=[0, 1],
    )
    parser.add_argument(
        "--run_zeroshot",
        type=int,
        default=1,
        help="Run zeroshot",
        choices=[0, 1],
    )
    parser.add_argument(
        "--run_fewshot",
        type=int,
        default=1,
        help="Run fewshot",
        choices=[0, 1],
    )
    parser.add_argument(
        "--run_zeroshot_decomp",
        type=int,
        default=0,
        help="Run zero shot decomp",
        choices=[0, 1],
    )
    parser.add_argument(
        "--overwrite_boost_exs",
        type=int,
        default=0,
        help="Overwrite boost examples",
        choices=[0, 1],
    )
    parser.add_argument(
        "--overwrite_data",
        type=int,
        default=0,
        help="Overwrite saved data examples",
        choices=[0, 1],
    )
    # Manifest
    parser.add_argument(
        "--client_name",
        type=str,
        default="huggingface",
        help="Client name manifest",
        choices=["huggingface", "openai", "ai21"],
    )
    parser.add_argument(
        "--client_engine",
        type=str,
        default=None,
        help="Client engine manifest. Only used for openai/ai21",
        choices=["davinci"],
    )
    parser.add_argument(
        "--client_connection",
        type=str,
        default="http://127.0.0.1:5001",
        help="Client connection str",
    )
    parser.add_argument(
        "--cache_connection",
        type=str,
        default="/home/manifest/final_runs.sqlite",
        help="Cache connection str",
    )
    parser.add_argument(
        "--overwrite_manifest",
        type=int,
        default=0,
        help="Overwrite manifest",
        choices=[0, 1],
    )
    return parser.parse_args()

class Decomposition:
    def __init__(self, task_name, data_dir, val_split="validation"):
        self.task_name = task_name
        self.data_dir = data_dir
        self.val_split = val_split

    def read_data(self, save_dir, overwrite_data):
        save_data = Path(f"{save_dir}/{self.task_name}/data.feather")
        if not save_data.exists() or overwrite_data:
            test_data = pd.read_feather(f"{self.data_dir}/{self.val_split}.feather")
            test_data.to_feather(f"{save_data}")
        else:
            print(f"Reading test data from {save_data}")
            test_data = pd.read_feather(save_data)

        save_data = Path(f"{save_dir}/{self.task_name}/train_data.feather")
        if not save_data.exists() or overwrite_data:
            train_data = pd.read_feather(f"{self.data_dir}/train.feather")
        else:
            print(f"Reading train data from {save_data}")
            train_data = pd.read_feather(save_data)
        print(f"Test Data Size: {len(test_data)}")
        print(f"Train Data Size: {len(train_data)}")
        return test_data, train_data

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        return train_data.sample(k_shot)

    def get_boost_decomp_examples(self, train_data, boost_i=0):
        """Get boost examples"""
        raise NotImplementedError()

    def zero_few_baseline(
        self, test_data, few_shot_df, manifest, overwrite_manifest, do_few_shot=True
    ):
        """Zero and few shot baseline"""
        raise NotImplementedError()

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        """Decomposition run"""
        raise NotImplementedError()

    def merge_boosted_preds(self, boosted_preds, all_boost_train_preds, train_labels, exp_log, expt_log_train, indecisive_ans=None):
        """Merge boosted preds"""
        if isinstance(boosted_preds, list):
            boosted_preds = np.array(boosted_preds)
        if isinstance(all_boost_train_preds, list):
            all_boost_train_preds = np.array(all_boost_train_preds)
        if isinstance(train_labels, list):
            train_labels = np.array(train_labels)

        uniq = np.unique(boosted_preds)
        pred_map = {}
        if "yes" in uniq:
            pred_map = {"yes": 1, "no": -1, "neither": 0}
        elif "true" in uniq:
            pred_map = {"true": 1, "false": -1, "neither": 0}
        elif "positive" in uniq:
            pred_map = {"positive": 1, "negative": -1, "neutral": 0}
        pred_map_inv = {v:k for k,v in pred_map.items()}
        use_pred_map = False
        if all(p.lower() in pred_map for p in uniq):
            use_pred_map = True
        if use_pred_map:
            # Cast to integers
            boosted_preds = np.array([[pred_map[p.lower()] for p in preds] for preds in boosted_preds])
            all_boost_train_preds = np.array(
                [[pred_map[p.lower()] for p in preds] for preds in all_boost_train_preds]
            )
            train_labels = np.array([pred_map[p.lower()] for p in train_labels])
            if indecisive_ans:
                indecisive_ans = pred_map[indecisive_ans.lower()]
        
        # Take majority vote
        preds_test = []
        for i, voter_preds in enumerate(boosted_preds):
            most_common = Counter(voter_preds).most_common(1)[0]
            if indecisive_ans and len(voter_preds) > 1 and most_common[1] == 1:
                majority_vote_pred = indecisive_ans
            else:
                majority_vote_pred = most_common[0]
            if use_pred_map:
                majority_vote_pred = pred_map_inv[majority_vote_pred]
            preds_test.append(majority_vote_pred)
            exp_log[i]["pred"] = majority_vote_pred

        # Take majority vote
        preds_train = []
        for i, voter_preds in enumerate(all_boost_train_preds):
            most_common = Counter(voter_preds).most_common(1)[0]
            if indecisive_ans and len(voter_preds) > 1 and most_common[1] == 1:
                majority_vote_pred = indecisive_ans
            else:
                majority_vote_pred = most_common[0]
            if use_pred_map:
                majority_vote_pred = pred_map_inv[majority_vote_pred]
            preds_train.append(majority_vote_pred)
            expt_log_train[i]["pred"] = majority_vote_pred
        return preds_test

    def run(self, args):
        print(json.dumps(vars(args), indent=4))

        random.seed(args.seed)
        np.random.seed(args.seed)
        save_path = Path(f"{args.save_dir}/{self.task_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        data_test, data_train = self.read_data(args.save_dir, bool(args.overwrite_data))
        # Subsample train for boost exps
        if args.boost_train_examples >= 0:
            boost_data_train = data_train.head(min(len(data_train), args.boost_train_examples))
        else:
            boost_data_train = data_train
        # Reset indexes for enumerations
        boost_data_train = boost_data_train.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)
        data_train = data_train.reset_index(drop=True)
        num_run = (
            min(args.num_run, len(data_test)) if args.num_run > 0 else len(data_test)
        )
        save_results = True
        if num_run != len(data_test):
            print("Using {} rows".format(num_run))
            data_test = data_test.iloc[:num_run]
            save_results = False

        runner, model_name = get_manifest_session(
            client_name=args.client_name,
            client_engine=args.client_engine,
            client_connection=args.client_connection,
            cache_connection=args.cache_connection,
        )

        model_name = model_name.replace("/", "_")
        print("Model name:", model_name)

        # Read in few shot examples
        few_shot_path = save_path /f"{args.k_shot}_shot_examples.feather"
        if bool(args.overwrite_data) or not few_shot_path.exists():
            mini_df = self.get_few_shot_examples(data_train, args.k_shot)
            mini_df.reset_index().to_feather(few_shot_path)
        else:
            print(f"Reading few show examples from {few_shot_path}")
            mini_df = pd.read_feather(few_shot_path)

        # Read in few shot decomp examples - one data frame per decomp step
        boost_examples = []
        for i in range(args.num_boost):
            boost_examples_per_step = []
            # Get all steps
            boost_examples_paths = list(save_path.glob(f"boost_examples_{i}_step*.feather"))
            if bool(args.overwrite_boost_exs) or not boost_examples_paths or not all(p.exists() for p in boost_examples_paths):
                boost_df_steps = self.get_boost_decomp_examples(data_train, boost_id=i)
                if not isinstance(boost_df_steps, list) or not isinstance(
                    boost_df_steps[0], pd.DataFrame
                ):
                    raise ValueError("Must return list of dataframes, one per step")
                for step, boost_df in enumerate(boost_df_steps):
                    boost_df.reset_index().to_feather(save_path / f"boost_examples_{i}_step{step}.feather")
                    print(f"Saving boost examples to", save_path / f"boost_examples_{i}_step{step}.feather")
                    boost_examples_per_step.append(boost_df)
            else:
                for boost_examples_p in sorted(boost_examples_paths):
                    print(f"Reading boost examples from {boost_examples_p}")
                    boost_examples_per_step.append(pd.read_feather(boost_examples_p))
            boost_examples.append(boost_examples_per_step)

        today = datetime.datetime.today().strftime("%m%d%Y")

        # Default metrics
        metric_zero = -1.0
        metric_few = -1.0
        metric_decomposed = -1.0
        metric_decomposed_by_boost = []
        metric_zeroshot_decomposed = -1.0

        if bool(args.run_zeroshot):
            # Zero Shot
            run_name = f"{model_name}_0shot"
            exp_zero, metric_zero = self.zero_few_baseline(
                test_data=data_test,
                few_shot_df=mini_df,
                manifest=runner,
                overwrite_manifest=args.overwrite_manifest,
                do_few_shot=False,
            )
            if save_results:
                save_log(self.task_name, run_name, exp_zero, args.save_dir)

        if bool(args.run_fewshot):
            # Few Shot
            run_name = f"{model_name}_{args.k_shot}shot"
            exp_few, metric_few = self.zero_few_baseline(
                test_data=data_test,
                few_shot_df=mini_df,
                manifest=runner,
                overwrite_manifest=args.overwrite_manifest,
                do_few_shot=True,
            )
            if save_results:
                save_log(self.task_name, run_name, exp_few, args.save_dir)

        if bool(args.run_decomp):
            # Decomp
            run_name = f"{model_name}_decomposed_{today}"
            exp_decomposed, exp_decomposed_train, metric_decomposed, metric_decomposed_by_boost = self.run_decomposed_prompt(
                test_data=data_test, boost_data_train=boost_data_train, boost_dfs=boost_examples, manifest=runner, overwrite_manifest=args.overwrite_manifest
            )
            if save_results:
                save_log(
                    self.task_name,
                    run_name,
                    exp_decomposed,
                    args.save_dir
                )
                if exp_decomposed_train:
                    save_log(
                        self.task_name,
                        f"{run_name}_train",
                        exp_decomposed_train,
                        args.save_dir
                    )

        # Zero shot decomp
        exp_zeroshot_decomposed = []
        if bool(args.run_zeroshot_decomp):
            run_name = f"{model_name}_decomposed_0shot_{today}"
            (
                exp_zeroshot_decomposed,
                exp_zeroshot_decomposed_train,
                metric_zeroshot_decomposed,
                _,
            ) = self.run_decomposed_prompt(
                test_data=data_test, boost_data_train=boost_data_train, boost_dfs=[[pd.DataFrame() for _ in range(len(boost_examples[0]))]], manifest=runner, overwrite_manifest=args.overwrite_manifest,
            )
            if save_results and len(exp_zeroshot_decomposed) > 0:
                save_log(
                    self.task_name,
                    run_name,
                    exp_zeroshot_decomposed,
                    args.save_dir,
                )
                if exp_zeroshot_decomposed_train:
                    save_log(
                    self.task_name,
                    f"{run_name}_train",
                    exp_zeroshot_decomposed_train,
                    args.save_dir,
                )
        print("Accuracy Zero Shot", metric_zero)
        print("Accuracy Few Shot", metric_few)
        if len(metric_decomposed_by_boost) > 0:
            print("Accuracy by Boost Set Decomposed", metric_decomposed_by_boost)
            print("Accuracy by Boost Set Decomposed Average", np.mean(metric_decomposed_by_boost))
        print("Accuracy Boost Decomposed", metric_decomposed)
        if len(exp_zeroshot_decomposed) > 0:
            print("Accuracy Zero Shot Decomposed", metric_zeroshot_decomposed)
        
        metrics = {
            "model_name": model_name,
            "task_name": self.task_name,
            "today": today,
            "zero_shot": metric_zero,
            "few_shot": metric_few,
            "decomposed": metric_decomposed,
            "decomposed_by_boost": metric_decomposed_by_boost,
            "decomposed_by_boost_avg": np.mean(metric_decomposed_by_boost),
            "zero_shot_decomposed": metric_zeroshot_decomposed,
        }
        output_metrics = Path(args.output_metrics_file)
        output_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(output_metrics, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"Saved metrics to {output_metrics}")
        print(f"Saved final data to", Path(args.save_dir) / self.task_name)
