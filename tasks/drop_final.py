#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset

from decomposition import Decomposition, get_args
from utils import get_response, text_f1, InputOutputPrompt, load_hf_data

extract = InputOutputPrompt(
    input_formatter=lambda x: f"Context: {x['context']}\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["context", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Answer the question. If there is no evidence in the context, return \"Unknown\".\n\n"
)

extract_examples = [
    pd.DataFrame([
        {
            "context": "According to Biraben, the plague was present somewhere in Europe in every year between 1346 and 1671",
            "question": "Where was the plague present?",
            "answer": "somewhere in Europe"
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "What's one factor in increasing self-esteem?",
            "answer": "Unknown"
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "What is another name for anti-matter?",
            "answer": "Unknown"
        }
    ]),
    pd.DataFrame([
        {
            "context": "According to Biraben, the plague was present somewhere in Europe in every year between 1346 and 1671",
            "question": "Where was the plague present?",
            "answer": "somewhere in Europe"
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "What is another name for anti-matter?",
            "answer": "Unknown"
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "What's one factor in increasing self-esteem?",
            "answer": "Unknown"
        },
    ]),
    pd.DataFrame([
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "What's one factor in increasing self-esteem?",
            "answer": "Unknown"
        },
        {
            "context": "According to Biraben, the plague was present somewhere in Europe in every year between 1346 and 1671",
            "question": "Where was the plague present?",
            "answer": "somewhere in Europe"
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "What is another name for anti-matter?",
            "answer": "Unknown"
        }
    ]),

]

prefix_select_zeroshot = """Answer the question. If there is no evidence in the context, return "Unknown".\n\n"""


class DropDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def read_data(self, save_dir, overwrite_data):
        return load_hf_data(save_dir, self.task_name, self.val_split, "drop", overwrite_data)

    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            extract_examples[boost_id],
        ]

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = []
        for x in train_data.answers_spans:
            if len(x['spans']) == 0:
                labels.append("unknown")
            else:
                labels.append(x['spans'][0])  
        train_data['expanded_labels'] = labels

        labels = ["unknown"] + list(set(labels))

        num_per_class = int(np.ceil(k_shot / len(labels)))
        print(f"Selecting {num_per_class} examples per class.")

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1
            sub_df = train_data[train_data["expanded_labels"] == label].sample(num_per_class)
            dfs.append(sub_df)
            total_in_context += num_per_class
            if total_in_context == k_shot:
                break
        mini_df = pd.concat(dfs)
        return mini_df

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
                entry = expt_log[ind]
                pred = entry["pred"]
                gold = entry["gold"]

            else:
                text = row.passage
                question = row.question
                if len(row.answers_spans["spans"]) == 0:
                    label = "unknown"
                else:
                    label = row.answers_spans["spans"][0]
                gold = label.lower()

                icl_str = ""
                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        input = s_row.passage
                        s_question = s_row.question
                        if len(s_row.answers_spans["spans"]) == 0:
                            label = "unknown"
                        else:
                            label = s_row.answers_spans["spans"][0]
                        icl_str += f"Passage: {input}\nQuestion: {s_question}\nAnswer: {label}\n\n"

                prompt = (
                    icl_str
                    + "Passage: {text:}\nQuestion: {question:}"
                    + "\nAnswer:"
                )
                pmp = prompt.format(text=text, question=question)
                if i == 0:
                    print(prompt.format(text=text, question=question))

                raw_answer = get_response(
                    pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=10,
                    stop_token="\n\n",
                )
                pred = raw_answer.strip("\n").strip().lower()
                entry = {
                    "ind": ind,
                    "example": text,
                    "base_prompt": pmp,
                    "raw_answer": raw_answer,
                    "pred": pred,
                    "gold": gold,
                }
                expt_log[ind] = entry

            preds.append(pred)
            labels.append(gold)
        metric = text_f1(preds=preds, golds=labels)
        return expt_log, metric

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest, run_limit=1)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train)
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            individual_accuracies.append(text_f1(preds=[p[i] for p in all_boost_preds], golds=labels))
        metric = text_f1(preds=preds, golds=labels)
        return expt_log, expt_log_train, metric, individual_accuracies

    def _run_decomp_single_data(self, test_data, boost_dfs, manifest, overwrite_manifest, run_limit=-1):
        expt_log = {}
        all_boost_preds = []
        labels = []

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            if i == run_limit:
                break

            text = row.passage
            question = row.question
            if len(row.answers_spans["spans"]) == 0:
                label = "unknown"
            else:
                label = row.answers_spans["spans"][0]
            gold = label.lower()
            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                prompt_suffix = extract(boost_examples[0])
                prompt = (
                    prompt_suffix + "\n\nContext: {text:}\nQuestion: {question:}\nAnswer:"
                )
                pmp = prompt.format(text=text, question=question)
                if i == 0:
                    print(pmp)

                raw_answer = get_response(
                    pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=50,
                    stop_token="\n\n",
                )
                pred = raw_answer.split("\n")[0].replace('"', "").strip().lower()
                # Single list pmp for one step decomp
                prompts_across_boost.append([pmp])
                preds_across_boost.append(pred)
            
            entry = {
                "ind": ind,
                "example": text,
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "gold": gold,
            }
            expt_log[ind] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(gold)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    task_name = "drop"
    data_dir = "drop"
    wic = DropDecomp(task_name, data_dir)
    wic.run(args)


if __name__ == "__main__":
    main()
