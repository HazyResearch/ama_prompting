#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import random
from datasets import load_dataset

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

##############################################################################################################################
# All prompts
summarize = InputOutputPrompt(
    input_formatter=lambda x: f"Product: {x['product']}",
    output_formatter=lambda x: f"Summarize: the product \"Product\": {x['summarize']}",
    required_keys=["product", "summarize"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Summarize the product.\n\n"
)
summarize_examples = [
    pd.DataFrame([
        {
            "product": "Was unsure when I purchased the DVD what to expect.  With real joy I can say that it was worth every cent and I have already watched it several times. The Storyline kept me interested.",
            "summarize": "The product is a DVD."
        },
        {
            "product": "These are the best headphones I've ever owned. I recently purchased a replacement pair, as my original set died after several years of intensive use.",
            "summarize": "The product is headphones."
        },
        {
            "product": "So these tights are tighter than most tights I own and when I take these off, they leave my legs feeling like they've been squeezed to death.",
            "summarize": "The product is tights."
        }
    ]),
    pd.DataFrame([
        {
            "product": "This bra is extremely comfortable, affordable and pretty too! My only complaint, and the reason for 4 stars is that the straps can't be adjusted very much.",
            "summarize": "The product is a bra."
        },
        {
            "product": "1/8/10 Have been using this drill and am very pleased. It has tons of torque and the handle comes in handy.",
            "summarize": "The product is a drill."
        },
        {
            "product": "I have used the Sanford highlighters for close to 20 years. there are nice. They are almost a disaster when highlighting textbooks.",
            "summarize": "The product is a highlighter."
        }
    ]),
    pd.DataFrame([
        {
            "product": "bought a pack of these at a b&m; store, and you'd think pens are pens... especially if you paid a measly $2 for a 12 pack. But negative. These pens I bought were dry.",
            "summarize": "The product is a pen."
        },
        {
            "product": "I get a lot of grease on my guitar from lotion, sweat, fingerprints, dust, what have you; I take some of this, spray it on a cloth, give it some elbow grease, and my guitars are shiny as the day it was made.",
            "summarize": "The product is a guitar."
        },
        {
            "product": "I purchased this sander nearly one year ago and can't say I have any complaints about it. The dust collection works surprisingly well, though if the bag isn't pushed in all",
            "summarize": "The product is a sander."
        }
    ]),
    pd.DataFrame([
        {
            "product": "I have 7 guitars in my cramped little bedroom studio and I quickly ran out of space to hold them easily. Floor space is dominated by my desk and drum set and I wanted my guitars to be out of the way and safe so they didn't get tripped over or dinged.",
            "summarize": "The product is guitars."
        },
        {
            "product": "This is a beautifully constructed book. The circus atmosphere is rich and detailed, and it's redolent of its time period. The images are strong and the pace, while not fast, is stately -- perhaps the way an elephant moves??",
            "summarize": "The product is a book."
        },
        {
            "product": "I was looking for decent Levi's for a few years and Amazon had them!!! I wanted the stiff unwashed jeans because they last quite a few years.",
            "summarize": "The product is jeans."
        }
    ]),
    pd.DataFrame([
        {
            "product": "I get a lot of grease on my guitar from lotion, sweat, fingerprints, dust, what have you; I take some of this, spray it on a cloth, give it some elbow grease, and my guitars are shiny as the day it was made.",
            "summarize": "The product is a guitar."
        },
        {
            "product": "This bra is extremely comfortable, affordable and pretty too! My only complaint, and the reason for 4 stars is that the straps can't be adjusted very much.",
            "summarize": "The product is a bra."
        },
        {
            "product": "The parts of this book that dealt with the main character in old age, were very insightful and I enjoyed that. But quite honestly had I known that it would detail the abuse of the circus animals and then also the detailed sex acts I would never had purchased this.",
            "summarize": "The product is a book."
        }
    ])
]

categorize = InputOutputPrompt(
    input_formatter=lambda x: f"Product: {x['product']}\nSummary: {x['summary']}",
    output_formatter=lambda x: f"The summary \"Summary\" fits \"Category\": {x['category']}",
    required_keys=["product", "summary", "category"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Pick the correct category for the product.\n\n\"Categories\":\n- Amazon Instant Video\n- Books\n- Clothing Shoes and Jewelry\n- Electronics\n- Kindle Store\n- Movies and TV\n- Musical Instruments\n- Office Products\n- Tools and Home Improvement\n\n"
)
categorize_examples = [
    pd.DataFrame([
        {
            "product": "Was unsure when I purchased the DVD what to expect.  With real joy I can say that it was worth every cent and I have already watched it several times.  The Storyline kept me interested.",
            "summary": "The product is a DVD.",
            "category": "Amazon Instant Video"
        },
        {
            "product": "These are the best headphones I've ever owned. I recently purchased a replacement pair, as my original set died after several years of intensive use.",
            "summary": "The product is headphones.",
            "category": "Electronics"
        },
        {
            "product": "So these tights are tighter than most tights I own and when I take these off, they leave my legs feeling like they've been squeezed to death.",
            "summary": "The product is tights.",
            "category": "Clothing Shoes and Jewelry"
        }
    ]),
    pd.DataFrame([
        {
            "product": "This bra is extremely comfortable, affordable and pretty too! My only complaint, and the reason for 4 stars is that the straps can't be adjusted very much. ",
            "summary": "The product is a bra.",
            "category": "Clothing Shoes and Jewelry"
        },
        {
            "product": "1/8/10 Have been using this drill and am very pleased. It has tons of torque and the handle comes in handy. ",
            "summary": "The product is a drill.",
            "category": "Tools and Home Improvement"
        },
        {
            "product": "I have used the Sanford highlighters for close to 20 years. there are nice. They are almost a disaster when highlighting textbooks. ",
            "summary": "The product is a highlighter.",
            "category": "Office Products"
        }
    ]),
    pd.DataFrame([
        {
            "product": "bought a pack of these at a b&m; store, and you'd think pens are pens... especially if you paid a measly $2 for a 12 pack. But negative. These pens I bought were dry.",
            "summary": "The product is a pen.",
            "category": "Office Products"
        },
        {
            "product": "I get a lot of grease on my guitar from lotion, sweat, fingerprints, dust, what have you; I take some of this, spray it on a cloth, give it some elbow grease, and my guitars are shiny as the day it was made. ",
            "summary": "The product is a guitar.",
            "category": "Musical Instruments"
        },
        {
            "product": "I purchased this sander nearly one year ago and can't say I have any complaints about it. The dust collection works surprisingly well, though if the bag isn't pushed in all",
            "summary": "The product is a sander.",
            "category": "Tools and Home Improvement"
        }
    ]),
    pd.DataFrame([
        {
            "product": "I have 7 guitars in my cramped little bedroom studio and I quickly ran out of space to hold them easily. Floor space is dominated by my desk and drum set and I wanted my guitars to be out of the way and safe so they didn't get tripped over or dinged.",
            "summary": "The product is guitars.",
            "category": "Musical Instruments"
        },
        {
            "product": "This is a beautifully constructed book. The circus atmosphere is rich and detailed, and it's redolent of its time period. The images are strong and the pace, while not fast, is stately -- perhaps the way an elephant moves??",
            "summary": "The product is a book.",
            "category": 'Books',
        },
        {
            "product": "I was looking for decent Levi's for a few years and Amazon had them!!! I wanted the stiff unwashed jeans because they last quite a few years.",
            "summary": 'The product is jeans',
            "category": 'Clothing Shoes and Jewelry',
        }
    ]),
    pd.DataFrame([
        {
            "product": "I get a lot of grease on my guitar from lotion, sweat, fingerprints, dust, what have you; I take some of this, spray it on a cloth, give it some elbow grease, and my guitars are shiny as the day it was made.",
            "summary": "The product is a guitar.",
            "category": "Musical Instruments"
        },
        {
            "product": "The parts of this book that dealt with the main character in old age, were very insightful and I enjoyed that. But quite honestly had I known that it would detail the abuse of the circus animals and then also the detailed sex acts I would never had purchased this.",
            "summary": 'The product is a book.',
            "category": 'Books',
        },
        {
            "product": "This bra is extremely comfortable, affordable and pretty too! My only complaint, and the reason for 4 stars is that the straps can't be adjusted very much.",
            "summary": "The product is a bra.",
            "category": "Clothing Shoes and Jewelry"
        }
    ])
]
description_zeroshot="""
Pick the correct category for the product.
Categories:
- Amazon Instant Video
- Books
- Clothing Shoes and Jewelry
- Electronics
- Kindle Store
- Movies and TV
- Musical Instruments
- Office Products
- Tools and Home Improvement"""

class AmazonProduct(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            summarize_examples[boost_id],
            categorize_examples[boost_id],
        ]

    def read_data(self, save_dir, overwrite_data):
        random.seed(0)
        np.random.seed(0)
        save_data = Path(f"{save_dir}/{self.task_name}/data.feather")
        dataset = load_dataset('amazon')
        all_data = dataset['test']
        all_data = pd.DataFrame(all_data).sample(frac=1, random_state=0)
        if not save_data.exists() or overwrite_data:
            test_data = all_data.iloc[:int(len(all_data)*0.9)]
            test_data.reset_index().to_feather(f"{save_data}")
        else:
            print(f"Reading test data from {save_data}")
            test_data = pd.read_feather(save_data)

        save_data = Path(f"{save_dir}/{self.task_name}/train_data.feather")
        if not save_data.exists() or overwrite_data:
            train_data = all_data.iloc[int(len(all_data)*0.9):]
            train_data.reset_index().to_feather(f"{save_data}")
        else:
            print(f"Reading train data from {save_data}")
            train_data = pd.read_feather(save_data)

        print(f"Test Data Size: {len(test_data)}")
        print(f"Train Data Size: {len(train_data)}")
        return test_data, train_data

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = ['Clothing_Shoes_and_Jewelry', 'Tools_and_Home_Improvement', 'Office_Products', 'Amazon_Instant_Video', 'Musical_Instruments', 'Books', 'Electronics', 'Kindle_Store']
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
            text = row['text']
            gold = row['label']
            gold = gold.replace("_", " ").strip().replace(",", "")
            
            icl_str = f"{description_zeroshot}"
            if do_few_shot:
                for s_ind, s_row in few_shot_df.iterrows():
                    s_label = s_row['label'].replace("_", " ")
                    icl_str += f"\n\nProduct: {s_row['text']}\nCategory: {s_label}"

            prompt = f"{icl_str}\n\nProduct: {{text:}}\nCategory:"
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
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest, run_limit=-1)
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

        labels_clean = [l.replace("_", " ") for l in set(test_data['label'])]

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            text = row['text']
            gold = row['label']

            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                all_prompts = []
                prompt_suffix = summarize(boost_examples[0])
                summary_prompt = f"{prompt_suffix}\n\nProduct: {{text:}}\nSummarize: the product \"Product\":" 
                summary_pmp = summary_prompt.format(text=text)
                output = get_response(
                    summary_pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=25,
                )
                summary = output.split("\n")[0].split(":")[-1].strip("\n")
                all_prompts.append(summary_pmp)
                
                prompt_suffix = categorize(boost_examples[1])
                category_prompt = f"{prompt_suffix}\n\nProduct: {{text:}}\nSummary: {{summary:}}\nThe summary \"Summary\" fits \"Category\":"
                category_pmp = category_prompt.format(text=text, summary=summary)
                output = get_response(
                    category_pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=10,
                )
                all_prompts.append(category_pmp)
                if i == 0:
                    print(summary_pmp)
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
            expt_log[i] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(gold)
        return expt_log, all_boost_preds, labels

def main():
    args = get_args()
    args.num_boost = 3
    task_name = "amazon_products"
    data_dir = (
        f"{DATA_DIR}/amazon_products/"
    )
    if not Path(data_dir).exists():
        raise ValueError(
            f"Data directory {data_dir} does not exist."
            "Download from https://github.com/allenai/flex/blob/75d6d1cea66df2c8a7e3d429c6af5008ccf1544b/fewshot/hf_datasets_scripts/amazon/amazon.py"
        )
    decomp = AmazonProduct(task_name, data_dir, val_split="test")
    decomp.run(args)


if __name__ == "__main__":
    main()
