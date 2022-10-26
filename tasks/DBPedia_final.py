#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

summarize = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['passage']}",
    output_formatter=lambda x: f"Summarize: the passage \"Passage\": {x['summary']}",
    required_keys=["passage", "summary"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Summarize the passage.\n\n\"Categories\":\n- company\n- educational institution\n- artist\n- athlete\n- office holder\n- mean of transportation\n- building\n- natural place\n- village\n- animal\n- plant\n- album\n- film \n- written work\n\n"
)

summarize_examples = [
    pd.DataFrame([
        {
            "passage": "Personality and Mental Health -  Personality and Mental Health: Multidisciplinary Studies from Personality Dysfunction to Criminal Behaviour is a quarterly peer-reviewed academic journal published by Wiley-Blackwell on behalf of the Centre for Health and Justice.",
            "summary": "The passage is about a journal."
        },
        {
            "passage": "RNLB Mona (ON 775) -  RNLB Mona (ON 775) was a Watson Class lifeboat based at Broughty Ferry in Scotland that capsized during a rescue attempt with the loss of her entire crew of eight men. The Mona was built in 1935 and in her time saved 118 lives.",
            "summary": "The passage is about a lifeboat."
        },
        {
            "passage": "Sayonara mo Ienakatta Natsu -  Sayonara mo Ienakatta Natsu (さよならも言えなかった夏) is an album by Mikuni Shimokawa released on July 4 2007 by Pony Canyon.This album consists of eleven songs; several new songs and some songs which were previously released as singles.",
            "summary": "The passage is about a album."
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Personality and Mental Health -  Personality and Mental Health: Multidisciplinary Studies from Personality Dysfunction to Criminal Behaviour is a quarterly peer-reviewed academic journal published by Wiley-Blackwell on behalf of the Centre for Health and Justice.",
            "summary": "The passage is about a journal."
        },
        {
            "passage": "Sayonara mo Ienakatta Natsu -  Sayonara mo Ienakatta Natsu (さよならも言えなかった夏) is an album by Mikuni Shimokawa released on July 4 2007 by Pony Canyon.This album consists of eleven songs; several new songs and some songs which were previously released as singles.",
          "summary": "The passage is about a album."
        },
        {
            "passage": "RNLB Mona (ON 775) -  RNLB Mona (ON 775) was a Watson Class lifeboat based at Broughty Ferry in Scotland that capsized during a rescue attempt with the loss of her entire crew of eight men. The Mona was built in 1935 and in her time saved 118 lives.",
            "summary": "The passage is about a lifeboat."
        },
    ]),
    pd.DataFrame([
        {
            "passage": "Sayonara mo Ienakatta Natsu -  Sayonara mo Ienakatta Natsu (さよならも言えなかった夏) is an album by Mikuni Shimokawa released on July 4 2007 by Pony Canyon.This album consists of eleven songs; several new songs and some songs which were previously released as singles.",
            "summary": "The passage is about a album."
        },
        {
            "passage": "Personality and Mental Health -  Personality and Mental Health: Multidisciplinary Studies from Personality Dysfunction to Criminal Behaviour is a quarterly peer-reviewed academic journal published by Wiley-Blackwell on behalf of the Centre for Health and Justice.",
            "summary": "The passage is about a journal."
        },
        {
            "passage": "RNLB Mona (ON 775) -  RNLB Mona (ON 775) was a Watson Class lifeboat based at Broughty Ferry in Scotland that capsized during a rescue attempt with the loss of her entire crew of eight men. The Mona was built in 1935 and in her time saved 118 lives.",
            "summary": "The passage is about a lifeboat."
        },
    ])
]

categorize = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['passage']}\nSummary: {x['summary']}",
    output_formatter=lambda x: f"The summary \"Summary\" fits \"Category\": {x['category']}",
    required_keys=["passage", "summary", "category"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Pick one category for the following text.\n\n\"Categories\":\n- company\n- educational institution\n- artist\n- athlete\n- office holder\n- mean of transportation\n- building\n- natural place\n- village\n- animal\n- plant\n- album\n- film\n- written work\n\n"
)
categorize_examples = [
    pd.DataFrame([
        {
            "passage": "Personality and Mental Health -  Personality and Mental Health: Multidisciplinary Studies from Personality Dysfunction to Criminal Behaviour is a quarterly peer-reviewed academic journal published by Wiley-Blackwell on behalf of the Centre for Health and Justice.",
            "summary": "The passage is about a journal.",
            "category": "written work"
        },
        {
            "passage": "RNLB Mona (ON 775) -  RNLB Mona (ON 775) was a Watson Class lifeboat based at Broughty Ferry in Scotland that capsized during a rescue attempt with the loss of her entire crew of eight men. The Mona was built in 1935 and in her time saved 118 lives.",
            "summary": "The passage is about a lifeboat.",
            "category": "mean of transportation"
        },
        {
            "passage": "Sayonara mo Ienakatta Natsu -  Sayonara mo Ienakatta Natsu (さよならも言えなかった夏) is an album by Mikuni Shimokawa released on July 4 2007 by Pony Canyon.This album consists of eleven songs; several new songs and some songs which were previously released as singles.",
            "summary": "The passage is about a album.",
            "category": "album"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Personality and Mental Health -  Personality and Mental Health: Multidisciplinary Studies from Personality Dysfunction to Criminal Behaviour is a quarterly peer-reviewed academic journal published by Wiley-Blackwell on behalf of the Centre for Health and Justice.",
            "summary": "The passage is about a journal.",
            "category": "written work"
        },
        {
            "passage": "Sayonara mo Ienakatta Natsu -  Sayonara mo Ienakatta Natsu (さよならも言えなかった夏) is an album by Mikuni Shimokawa released on July 4 2007 by Pony Canyon.This album consists of eleven songs; several new songs and some songs which were previously released as singles.",
            "summary": "The passage is about a album.",
            "category": "album"
        },
        {
            "passage": "RNLB Mona (ON 775) -  RNLB Mona (ON 775) was a Watson Class lifeboat based at Broughty Ferry in Scotland that capsized during a rescue attempt with the loss of her entire crew of eight men. The Mona was built in 1935 and in her time saved 118 lives.",
            "summary": "The passage is about a lifeboat.",
            "category": "mean of transportation"
        },
    ]),
    pd.DataFrame([
        {
            "passage": "Sayonara mo Ienakatta Natsu -  Sayonara mo Ienakatta Natsu (さよならも言えなかった夏) is an album by Mikuni Shimokawa released on July 4 2007 by Pony Canyon.This album consists of eleven songs; several new songs and some songs which were previously released as singles.",
            "summary": "The passage is about a album.",
            "category": "album"
        },
        {
            "passage": "Personality and Mental Health -  Personality and Mental Health: Multidisciplinary Studies from Personality Dysfunction to Criminal Behaviour is a quarterly peer-reviewed academic journal published by Wiley-Blackwell on behalf of the Centre for Health and Justice.",
            "summary": "The passage is about a journal.",
            "category": "written work"
        },
        {
            "passage": "RNLB Mona (ON 775) -  RNLB Mona (ON 775) was a Watson Class lifeboat based at Broughty Ferry in Scotland that capsized during a rescue attempt with the loss of her entire crew of eight men. The Mona was built in 1935 and in her time saved 118 lives.",
            "summary": "The passage is about a lifeboat.",
            "category": "mean of transportation"
        },
    ])
]
description_zeroshot="""
Pick the correct category for the passage.
Categories:
- company
- educational institution
- artist
- athlete
- office holder
- mean of transportation
- building
- natural place
- village
- animal
- plant
- album
- film
- written work"""

class DBPediaDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            summarize_examples[boost_id],
            categorize_examples[boost_id],
        ]

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = sorted(set(train_data["targets_pretokenized"]))
        num_per_class = int(np.ceil(k_shot / len(labels)))
        print(f"Selecting {num_per_class} examples per class.")

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1
            sub_df = train_data[train_data["targets_pretokenized"] == label].sample(
                num_per_class
            )
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
            input = row["inputs_pretokenized"]
            body = input.split("written work. ")[-1]
            gold = row["targets_pretokenized"].strip().lower()
            icl_str = ""
            title = description_zeroshot
            if do_few_shot:
                for s_ind, s_row in few_shot_df.iterrows():
                    s_input = s_row.inputs_pretokenized
                    s_body = s_input.split("written work. ")[-1]
                    s_title = s_input.split("written work. ")[0] + "written work."
                    s_output = s_row.targets_pretokenized.strip()
                    icl_str += f"Passage: {s_body}\nCategory: {s_output}\n\n"

            icl_str = f"{title}\n\n{icl_str}"
            prompt = f"{icl_str}Passage: {{body:}}\nCategory:"
            pmp = prompt.format(body=body)
            if i == 0:
                print(pmp)

            output = get_response(
                pmp,
                manifest,
                overwrite=bool(overwrite_manifest),
                max_toks=10,
                stop_token="\n\n",
            )
            pred = output.strip().lower()
            entry = {
                "ind": ind,
                "example": input,
                "base_prompt": pmp,
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
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest, run_limit=1)
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
            if i == run_limit:
                break

            text = (
                row.inputs_pretokenized.strip("\n").split("written work.")[-1].strip()
            )
            gold = row.targets_pretokenized.strip().lower()

            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                all_prompts = []
                prompt_suffix = summarize(boost_examples[0])
                summarize_prompt = f"{prompt_suffix}\n\nPassage: {{text:}}\nSummarize: the passage \"Passage\":"
                summarize_pmp = summarize_prompt.format(text=text)
                output = get_response(
                    summarize_pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=50,
                )
                summary = output.split("\n")[0].split(":")[-1].strip("\n")
                prompt_suffix = categorize(boost_examples[1])
                category_prompt = f"{prompt_suffix}\n\nPassage: {{text:}}\nSummary: {{summary:}}\nThe summary \"Summary\" fits \"Category\":"
                category_pmp = category_prompt.format(text=text, summary=summary)
                output = get_response(
                    category_pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=15,
                )
                pred = output.split("\n")[0].strip().lower()

                all_prompts.append(summarize_pmp)
                all_prompts.append(category_pmp)
                if i == 0:
                    print(summarize_pmp)
                    print(category_pmp)
                prompts_across_boost.append(all_prompts)
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
    task_name = "dbpedia"
    data_dir = (
        f"{DATA_DIR}/P3/data_feather/dbpedia_14_pick_one_category_for_the_following_text"
    )
    decomp = DBPediaDecomp(task_name, data_dir, val_split="test")
    decomp.run(args)


if __name__ == "__main__":
    main()
