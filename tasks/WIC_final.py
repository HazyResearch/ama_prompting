#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
import pandas as pd

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

synonym = InputOutputPrompt(
    input_formatter=lambda x: f"{x['passage']}",
    output_formatter=lambda x: f"- {x['answer']}",
    required_keys=["passage", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Give synonyms of the word in the sentence.\n\n"
)
synonym_examples = [
    pd.DataFrame([
        {
            "passage": "In \"She heard the sound of voices in the hall.\", synonyms for the word \"sound\" are:",
            "answer": "noise",
        },
        {
            "passage": "In \"Enter the secret code.\", synonyms for the word \"code\" are:",
            "answer": "password",
        },
        {
            "passage": "In \"She acted in a play on Broadway\", synonyms for the word \"play\" are:",
            "answer": "show",
        },
    ]),
    pd.DataFrame([
        {
            "passage": "In \"She rode around the park on her cycle.\", synonyms for the word \"cycle\" are:",
            "answer": "bicycle",
        },
        {
            "passage": "In \"Don't keep everything bottled up.\", synonyms for the word \"bottled\" are:",
            "answer": "trapped inside",
        },
        {
            "passage": "In \"The present is like no other time.\", synonyms for the word \"present\" are:",
            "answer": "current moment",
        },
    ]),
    pd.DataFrame([
        {
            "passage": "In \"The movie was awful.\", synonyms for the word \"aweful\" are:",
            "answer": "bad and terrible",
        },
        {
            "passage": "In \"She is so beautiful.\", synonyms for the word \"beautiful\" are:",
            "answer": "pretty and gorgeous",
        },
        {
            "passage": "In \"It was quite cool out so she wore a jacket\", synonyms for the word \"cool\" are:",
            "answer": "cold and chilly",
        },
    ]),
    pd.DataFrame([
        {
            "passage": "In \"There are so many flies near the food.\", synonyms for the word \"flies\" are:",
            "answer": "bugs",
        },
        {
            "passage": "In \"Eat your noodles with a fork.\", synonyms for the word \"fork\" are:",
            "answer": "utensils",
        },
        {
            "passage": "In \"She and her husband went on a trip.\", synonyms for the word \"trip\" are:",
            "answer": "vacation",
        },
    ]),
    pd.DataFrame([
        {
            "passage": "In \"It was definitely a cry for help.\", synonyms for the word \"cry\" are:",
            "answer": "call",
        },
        {
            "passage": "In \"I watch all my students as they take their exams.\", synonyms for the word \"watch\" are:",
            "answer": "look at",
        },
        {
            "passage": "In \"The beginning of the book was fine, but the end was terrible.\", synonyms for the word \"beginning\" are:",
            "answer": "start",
        },
    ])
]

description = InputOutputPrompt(
    input_formatter=lambda x: f"Choices\n{x['choices']}\nFill the [MASK] with the correct \"Choice\": {x['sentence']}",
    output_formatter=lambda x: f"[MASK] is \"Choice\": {x['answer']}\n",
    required_keys=["choices", "sentence", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Select the correct choice for the blank.\n\n"
)
description_examples = [
    pd.DataFrame([
        {
            "choices": "1: noise\n2. in good condition",
            "sentence": "She heard the [MASK] of voices in the hall.",
            "answer": "noise",
        },
        {
            "choices": "1. not heavy\n2. sun rays",
            "sentence": "The [MASK] shined through the window.",
            "answer": "sun rays",
        },
        {
            "choices": "1. widespread\n2. commander of an army",
            "sentence": "The book is of [MASK] interest.",
            "answer": "widespread",
        },
    ])
]

class WICDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            synonym_examples[boost_id],
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
                        icl_str += f"{s_row['inputs_pretokenized']} {s_row['targets_pretokenized']}\n\n"

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
                answer = "".join(
                    [a for a in answer if a not in [".", ",", "?", ";", ":", "'", '"']]
                )

                is_yes = "yes" in answer.split()
                is_no = "no" in answer.split()
                pred = ""
                if is_yes and (not is_no):
                    pred = "yes"
                if is_no and (not is_yes):
                    pred = "no"

                gold = gold.strip().lower()
                pred = pred.strip().lower()
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

    def get_parts(self, text):
        parts = text.split("\n")
        sent1 = parts[0]
        sent2 = parts[1]
        word = parts[2].split("Question: Is the word ")[-1].split()[0]
        return word, sent1, sent2

    def clean_sentence(self, sentence):
        sentence = sentence.replace("2. ", "")
        sentence = sentence.replace("3. ", "")
        sentence = sentence.replace("\n\n", "")
        sentence = sentence.replace("A:", "")
        sentence = sentence.strip()
        sentence = sentence.split(".")[0]
        sentence = sentence.split("\n")[0]
        return sentence

    def get_sentences(
        self, all_constructors, all_boost_exs, sent, word, manifest, overwrite_manifest
    ):
        synonym = all_constructors[0]

        all_prompts = []
        # synonyms
        prompt_suffix = synonym(all_boost_exs[0])
        prompt_combined = f'{prompt_suffix}\n\nIn "{{sent:}}", synonyms for the word \"{{word:}}\" are: '
        all_prompts.append(prompt_combined.format(sent=sent, word=word))
        synonyms = get_response(
            prompt_combined.format(sent=sent, word=word),
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=10,
        )
        synonyms = synonyms.replace("- ", "").split("\n")
        synonyms = ", ".join([a for a in synonyms if a][0:1])

        # generate sentences
        quoted_sent = sent.split()
        sent = []
        for tok in quoted_sent:
            if tok.lower() == word.strip('"').lower():
                sent.append(f'"{tok}"')
            else:
                sent.append(tok)
        if sent:
            sent = " ".join(sent)
        else:
            sent = " ".join(quoted_sent)

        combined_definition = f"{synonyms}"
        sentences = []
        return combined_definition, sentences, all_prompts

    def pairwise_comparisons(
        self,
        description_constructor,
        boost_exs,
        def1,
        sentences_lst1,
        def2,
        sentences_lst2,
        word,
        manifest,
        overwrite_manifest,
    ):
        all_prompts = []
        # reconcile the result
        answer = ""
        if def1.strip() != def2.strip():
            answer = "No"
        else:
            answer = "Yes"
        return answer, all_prompts

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest)
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
            text = row["inputs_pretokenized"]
            gold = row["targets_pretokenized"]

            word, sent1, sent2 = self.get_parts(text)

            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                all_prompts = []

                def1, sentences_lst1, lst1_prompts = self.get_sentences(
                    [synonym], [boost_examples[0]], sent1, word, manifest, overwrite_manifest
                )
                def2, sentences_lst2, lst2_prompts = self.get_sentences(
                    [synonym], [boost_examples[0]], sent2, word, manifest, overwrite_manifest
                )

                pred, pred_prompts = self.pairwise_comparisons(
                    description,
                    boost_examples[-1],
                    def1,
                    sentences_lst1,
                    def2,
                    sentences_lst2,
                    word,
                    manifest,
                    overwrite_manifest,
                )
                all_prompts = lst1_prompts + lst2_prompts + pred_prompts
                if i == 0:
                    print("\n".join(all_prompts))
                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)
            entry = {
                "ind": ind,
                "example": text,
                "word": word,
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "sent1": sent1,
                "sent2": sent2,
                "def1": def1,
                "def2": def2,
                "gold": gold,
                "sentences_lst1": sentences_lst1,
                "sentences_lst2": sentences_lst2,
            }
            expt_log[ind] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(gold)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 5
    task_name = "super_glue_wic"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_wic_GPT_3_prompt/"
    decomp = WICDecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
