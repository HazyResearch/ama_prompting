#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

##############################################################################################################################
# All prompts
summarize = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['passage']}",
    output_formatter=lambda x: f"Summarize: the passage \"Passage\": {x['summarize']}",
    required_keys=["passage", "summarize"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Summarize the passage.\n\n"
)
summarize_examples = [
    pd.DataFrame([
        {
            "passage": "China overtakes United States as top destination for foreign investment (AFP). AFP - China overtook the United States as a top global destination for foreign direct investment (FDI) in 2003 while the Asia-Pacific region attracted more investment than any other developing region, a UN report said.",
            "summarize": "The passage is about foreign direct investment."
        },
        {
            "passage": "Colangelo resigns as CEO of D-Backs. Jerry Colangelo has resigned his position as chief executive officer of the Arizona Diamondbacks, effective immediately, handing the reins of the organization to CEO Elect Jeff Moorad.",
            "summarize": "The passage is about the Arizona Diamondbacks."
        },
        {
            "passage": "3 injured in plant fire in Japan. TOKYO, Aug. 20 (Xinhuanet) -- Fire broke out Friday at a tire plant belonging to Bridgestone Corp. in Amagi, western Fukuoka Prefecture of Japan, leaving 13 people injured.",
            "summarize": "The passage is about a plant fire."
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Abramovich faces rich list challenge. Lakshmi Mittal, the Indian-born steel magnate, yesterday staked a claim to overtake Roman Abramovich as Britain's richest man with a 10bn deal to create the world's largest steelmaker.",
            "summarize": "The passage is about a 10bn deal."
        },
        {
            "passage": "U.N. Deadlocks on Cloning Ban. The United Nations abandons efforts to ban all cloning and opts for a non-binding resolution. It's a blow to President Bush's efforts to push a ban and a victory for embryonic stem cell researchers. By Kristen Philipkoski",
            "summarize": "The passage is about stem cell research."
        },
        {
            "passage": "Tennis: Serena Williams Reaches Finals of China Open. Top seed Serena Williams of the United States has powered her way into the finals of the China Open tennis tournament in Beijing with a straight sets (6-2, 6-3) victory over fourth-seeded Vera Zvonareva of Russia.",
            "summarize": "The passage is about tennis."
        }
    ]),
    pd.DataFrame([
        {
            "passage": "San Francisco at Atlanta, 1:05 PM. ATLANTA (Ticker) -- Rookie Noah Lowry looks to win his fourth straight decision when he starts for the San Francisco Giants in the finale of a four-game series with the Atlanta Braves.",
            "summarize": "The passage is about the San Francisco Giants."
        },
        {
            "passage": "Suffocation cited in most deaths. At least 84 Muslim protesters died, mostly from suffocation so severe their eyes bled, after being arrested and locked in army trucks following clashes with security forces in the south, officials said yesterday.",
            "summarize": "The passage is about Muslim protesters."
        },
        {
            "passage": "Merrill, UBS Up Apple Stock Estimates. As consumers start spending on Christmas, two brokerage houses raised their estimates on Apple Computer (AAPL) stock Monday to more than US $77, predicting.",
            "summarize": "The passage is about Apple Stock Estimates."
        }
    ]),
]

categorize = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['passage']}\nSummary: {x['summary']}",
    output_formatter=lambda x: f"The summary \"Summary\" fits \"Category\": {x['category']}",
    required_keys=["passage", "summary", "category"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Pick the correct category for the passage.\n\n\"Categories\":\n- World News\n- Sports\n- Business\n- Technology and Science\n\n"
)
categorize_examples = [
    pd.DataFrame([
        {
            "passage": "China overtakes United States as top destination for foreign investment (AFP). AFP - China overtook the United States as a top global destination for foreign direct investment (FDI) in 2003 while the Asia-Pacific region attracted more investment than any other developing region, a UN report said.",
            "summary": "The passage is about foreign direct investment.",
            "category": "Business"
        },
        {
            "passage": "Colangelo resigns as CEO of D-Backs. Jerry Colangelo has resigned his position as chief executive officer of the Arizona Diamondbacks, effective immediately, handing the reins of the organization to CEO Elect Jeff Moorad.",
            "summary": "The passage is the Arizona Diamondbacks.",
            "category": "Sports"
        },
        {
            "passage": "3 injured in plant fire in Japan. TOKYO, Aug. 20 (Xinhuanet) -- Fire broke out Friday at a tire plant belonging to Bridgestone Corp. in Amagi, western Fukuoka Prefecture of Japan, leaving 13 people injured.",
            "summary": "The passage is about a plant fire.",
            "category": "World News"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Abramovich faces rich list challenge. Lakshmi Mittal, the Indian-born steel magnate, yesterday staked a claim to overtake Roman Abramovich as Britain's richest man with a 10bn deal to create the world's largest steelmaker.",
            "summary": "The passage is about a 10bn deal.",
            "category": "Business"
        },
        {
            "passage": "U.N. Deadlocks on Cloning Ban. The United Nations abandons efforts to ban all cloning and opts for a non-binding resolution. It's a blow to President Bush's efforts to push a ban and a victory for embryonic stem cell researchers. By Kristen Philipkoski",
            "summary": "The passage is about stem cell research.",
            "category": "Technology and Science"
        },
        {
            "passage": "Tennis: Serena Williams Reaches Finals of China Open. Top seed Serena Williams of the United States has powered her way into the finals of the China Open tennis tournament in Beijing with a straight sets (6-2, 6-3) victory over fourth-seeded Vera Zvonareva of Russia.",
            "summary": "The passage is about tennis",
            "category": "Sports"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "San Francisco at Atlanta, 1:05 PM. ATLANTA (Ticker) -- Rookie Noah Lowry looks to win his fourth straight decision when he starts for the San Francisco Giants in the finale of a four-game series with the Atlanta Braves.",
            "summary": "The passage is about the San Francisco Giants.",
            "category": "Sports"
        },
        {
            "passage": "Suffocation cited in most deaths. At least 84 Muslim protesters died, mostly from suffocation so severe their eyes bled, after being arrested and locked in army trucks following clashes with security forces in the south, officials said yesterday.",
            "summary": "The passage is about Muslim protesters.",
            "category": "World News"
        },
        {
            "passage": "Merrill, UBS Up Apple Stock Estimates. As consumers start spending on Christmas, two brokerage houses raised their estimates on Apple Computer (AAPL) stock Monday to more than US $77, predicting",
            "summary": "The passage is about Apple Stock Estimates.",
            "category": "Business"
        }
    ]),
]
description_zeroshot="""
Pick the correct category for the passage.

Categories:
- World News
- Sports
- Business
- Technology and Science"""

label_dict = {
    0: 'World News', 
    1: 'Sports', 
    2: 'Business', 
    3: 'Technology and Science'
}

def format_data(df):
    # Pre-processing code from: https://github.com/tonyzhaozh/few-shot-learning
    sentences = df['Title'] + ". " + df['Description']
    sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in sentences]) # some basic cleaning
    labels = list(df['Class Index'])
    labels = [l - 1 for l in labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    return sentences, labels

class AGNews(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def read_data(self, save_dir, overwrite_data):
        save_data = Path(f"{save_dir}/{self.task_name}/data.feather")
        if not save_data.exists() or overwrite_data:
            test_data = pd.read_csv(f"{self.data_dir}/{self.val_split}.csv")
            test_sentences, test_labels = format_data(test_data)
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
            train_data = pd.read_csv(f"{self.data_dir}/train.csv")
            train_sentences, train_labels = format_data(train_data)
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
        return [
            summarize_examples[boost_id],
            categorize_examples[boost_id],
        ]

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = [0, 1, 2, 3]
        num_per_class = int(np.ceil(k_shot / len(labels)))
        print(f"Selecting {num_per_class} examples per class.")

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1
            sub_df = train_data[train_data["label"] == label].sample(
                num_per_class
            )
            dfs.append(sub_df)
            total_in_context += num_per_class
            if total_in_context == k_shot:
                break
        mini_df = pd.concat(dfs)
        print(mini_df.index)
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
            text = row['sentence']
            gold = label_dict[row['label']]
            gold = gold.replace("_", " ").strip().replace(",", "")

            icl_str = f"{description_zeroshot}"
            if do_few_shot:
                for s_ind, s_row in few_shot_df.iterrows():
                    s_label = label_dict[s_row['label']]
                    icl_str += f"\n\nPassage: {s_row['sentence']}\nCategory: {s_label}"

            prompt = f"{icl_str}\n\nPassage: {{text:}}\nCategory:"
            pmp = prompt.format(text=text)

            if i == 0:
                print(pmp)

            answer = get_response(
                pmp,
                manifest,
                overwrite=bool(overwrite_manifest),
                max_toks=10,
                stop_token="\n\n",
            )
            answer = answer.split("\n")
            answer = [a for a in answer if a]
            pred = ''
            if answer:
                pred = answer[0]
            pred = pred.replace("-", "").strip().replace(",", "")

            entry = {
                "ind": ind,
                "example": text,
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

        labels_clean = [v for k, v in label_dict.items()]

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            text = row['sentence']
            gold = label_dict[row['label']]

            if i == run_limit:
                break
            
            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                all_prompts = []
                prompt_suffix = summarize(boost_examples[0])
                summary_prompt = f"{prompt_suffix}\n\nPassage: {{text:}}\nSummarize: the passage \"Passage\":"
                summary_pmp = summary_prompt.format(text=text) 
                output = get_response(
                    summary_pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=40,
                )
                summary = output.split("\n")[0].split(":")[-1].strip("\n")
                all_prompts.append(summary_pmp)
                
                prompt_suffix = categorize(boost_examples[1])
                category_prompt = f"{prompt_suffix}\n\nPassage: {{text:}}\nSummary: {{summary:}}\nThe summary \"Summary\" fits \"Category\":"
                category_pmp = category_prompt.format(text=text, summary=summary)
                output = get_response(
                    category_pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=5,
                )
                all_prompts.append(category_pmp)
                if i== 0:
                    print(summary_pmp)
                    print("\n---------------------------------------\n")
                    print(category_pmp)
                answer = output.split("\n")[0].strip().lower()
                answer = answer.replace("-", "").strip()
                gold = gold.replace("_", " ").strip().lower()
                pred = answer
                for label in labels_clean:
                    if label.lower() in answer.lower():
                        pred = label.lower()
                        break
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
    args.num_boost = 3
    task_name = "agnews"
    data_dir = f"{DATA_DIR}/agnews/"
    if not Path(data_dir).exists():
        raise ValueError(f"Data directory {data_dir} does not exist. Download AGNews from https://github.com/tonyzhaozh/few-shot-learning.")
    decomp = AGNews(task_name, data_dir, val_split="test")
    decomp.run(args)


if __name__ == "__main__":
    main()
