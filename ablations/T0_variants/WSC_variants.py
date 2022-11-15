#!/usr/bin/env python
# coding: utf-8
import os
from tqdm.auto import tqdm
import pandas as pd

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt
from collections import defaultdict, Counter

class RTEDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, data_train, boost_id):
        lst = [
            'super_glue_wsc.fixed_GPT_3_Style',
            'super_glue_wsc.fixed_Who_or_what_is_are',
            'super_glue_wsc.fixed_p_is_are_r',
            'super_glue_wsc.fixed_by_p_they_mean',
            'super_glue_wsc.fixed_replaced_with',
            'super_glue_wsc.fixed_in_other_words',
            'super_glue_wsc.fixed_I_think_they_mean',
            'super_glue_wsc.fixed_does_p_stand_for',
            'super_glue_wsc.fixed_does_the_pronoun_refer_to',
            'super_glue_wsc.fixed_the_pronoun_refers_to'
        ]
        file_path = lst[boost_id]
        print(f"FILE PATH: {file_path}")
        train_data = pd.read_feather(f"{DATA_DIR}/P3/data_feather/{file_path}/train.feather")
        val_data = pd.read_feather(f"{DATA_DIR}/P3/data_feather/{file_path}/validation.feather")
        return [
            train_data, 
            val_data
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

        labels_names = set(test_data["targets_pretokenized"])
        labels_names = [l.lower().strip() for l in labels_names]

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            if ind in expt_log:
                pred = entry["pred"]
                gold = entry["gold"]
            else:
                text = row["inputs_pretokenized"]
                gold = row["targets_pretokenized"]

                icl_str = ""
                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        icl_str += f"{s_row['inputs_pretokenized']} {s_row['targets_pretokenized']}\n\n\n"

                text = row["inputs_pretokenized"]
                text = text.replace("True, False, or Neither?", "")
                text = text + ". True, False, or Neither?"
                gold = row["targets_pretokenized"]
                prompt = f"{icl_str}{{text:}}"
                pmp = prompt.format(text=text)
                if i == 0:
                    print(pmp)

                raw_answer = get_response(
                    pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=30,
                )
                answer = raw_answer.strip().lower()
                answer = answer.split("\n")
                answer = [a for a in answer if a]
                answer = [
                    a
                    for a in answer
                    if any(l.lower() in a.lower() for l in labels_names)
                ]
                if answer:
                    answer = answer[0]
                else:
                    answer = ""
                answer = "".join(
                    [a for a in answer if a not in [".", ",", "?", ";", ":", "'", '"']]
                )
                is_yes = "true" in answer.split()
                is_no = "false" in answer.split()
                is_maybe = "neither" in answer.split() or "unknown" in answer.split() or "maybe" in answer.split()
                pred = "Neither"
                if is_yes and (not is_maybe and not is_no):
                    pred = "True"
                if is_no and (not is_maybe and not is_yes):
                    pred = "False"
                if is_maybe and (not is_no and not is_yes):
                    pred = "Neither"
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

        report = classification_report(labels, preds, output_dict=True)
        return expt_log, report["accuracy"]

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest, do_train=0)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest, do_train=1)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train)
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            individual_accuracies.append(classification_report(labels, [p[i] for p in all_boost_preds], output_dict=True)["accuracy"])
        report = classification_report(labels, preds, output_dict=True)
        return expt_log, expt_log_train, report["accuracy"], individual_accuracies

    def _run_decomp_single_data(self, test_data, boost_dfs, manifest, overwrite_manifest, do_train=-1):
        expt_log = {}
        all_boost_preds = []
        labels = []

        label_name_to_maping = {
            "always": "true", 
            "never": "false", 
            "sometimes": 'neither',
            "true": "true", 
            "false": "false", 
            "neither": 'neither',
            "no": "false",
            "yes": "true",
            "maybe": "neither",
            "unknown": "neither",
            "inconclusive": "neither",
            "impossible": "false",
            "possible": "neither",
            "guaranteed": "true",
        }

        prompts_across_boost = defaultdict(list)
        preds_across_boost = defaultdict(list)
        for boost_num, boost_examples in enumerate(boost_dfs):
            if do_train:
                data = boost_examples[0].iloc[:1000]
            elif not do_train:
                data = boost_examples[1]
            else:
                raise ValueError("Unsupported value for do train.")

            for i, (ind, row) in tqdm(enumerate(data.iterrows()), total=len(data)):
                input = row["inputs_pretokenized"]
                gold = row["targets_pretokenized"]
                all_prompts = []
                raw_answer = get_response(
                    input,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=30,
                )
                all_prompts.append(input)
                answer = raw_answer.lower()
                if answer not in label_name_to_maping:
                    pred = 'neither'
                    print(answer)
                else:
                    pred = label_name_to_maping[answer]
                prompts_across_boost[i].append(all_prompts)
                preds_across_boost[i].append(pred)
                if gold.lower() not in label_name_to_maping:
                    import pdb;
                    pdb.set_trace()

        for i, (ind, row) in enumerate(data.iterrows()):
            label = row["targets_pretokenized"].lower()
            entry = {
                "ind": ind,
                "prompts": prompts_across_boost[i],
                "preds_boost": preds_across_boost[i],
                "example": row['inputs_pretokenized'],
                "gold": label,
            }
            expt_log[ind] = entry
            all_boost_preds.append(preds_across_boost[ind])
            labels.append(label_name_to_maping[label]) 
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 10
    task_name = "wsc_t0_variants"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_wsc.fixed_GPT_3_Style/"
    decomp = RTEDecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
