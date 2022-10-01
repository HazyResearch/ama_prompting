#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

##############################################################################################################################
# All prompts
questioner_prompt = InputOutputPrompt(
    input_formatter=lambda x: f"Statement: {x['statement']}",
    output_formatter=lambda x: f"Question: {x['question']}",
    required_keys=["question", "statement"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Rewrite the statement as a yes/no question.\n\n"
)
questioner_prompt_examples = [
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
    ]),
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
    ]),
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
    ]),
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
    ]),
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
    ]),
]

extraction_qa = InputOutputPrompt(
    input_formatter=lambda x: f"Context: {x['context']}\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["context", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Answer the question. If there is no evidence in the context, return \"Unknown\".\n\n"
)
extraction_qa_examples = [
    pd.DataFrame([
        {
            "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
            "question": "Based on the context, Did the plague affect people in Europe?",
            "answer": "yes, people in Italy, Europe",
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
            "answer": "unknown",
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "Based on the context, Is anti-matter made of electrons? ",
            "answer": "Unknown",
        },
    ]),
    pd.DataFrame([
        {
            "context": "According to Biraben, the plague was present somewhere in Italy only between 1346 and 1671, and not after that.",
            "question": "Based on the context, Was the plague present in Italy during the 2000s?",
            "answer": "No, it was present between 1346 and 1671"
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "Based on the context, Is anti-matter made of electrons? ",
            "answer": "Unknown"
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
            "answer": "unknown"
        }
    ]),
    pd.DataFrame([
        {
            "context": "Jenna's 10th birthday was yesterday evening and at least 10 of her friends attended the party.",
            "question": "Based on the context, Did 10 friends attend Jenna's party?",
            "answer": "Unknown"
        },
        {
            "context": "The bullies attacked John when he was walking through the elementary school parking lot and then got sent to the teacher's office.",
            "question": "Based on the context, Did the bullies attack John in the teacher's office?",
            "answer": "No, parking lot"
        },
        {
            "context": "WISS discovered a new monkey disease occurring in a remote tribe in the Amazon rainforrest.",
            "question": "Based on the context, Did WISS discover a new monkey species?",
            "answer": "No, a new monkey disease"
        }
    ]),
    pd.DataFrame([
        {
            "context": "When Judy and Jack went to school, they got in trouble with their teacher for being late. I didn't think it was very fair.",
            "question": "Based on the context, Did she think it was fair?",
            "answer": "No"
        },
        {
            "context": "If inflation is occurring, leading to higher prices for basic necessities such as gas by 2 dollars. Do you think that inflation is good for society?",
            "question": "Based on the context, Is inflation good for society?",
            "answer": "Unknown"
        },
        {
            "context": "Put yourself out there. The more time you spend dating and socializing, the more likely you will find a boyfriend you like.",
            "question": "Based on the context, Does socializing help you find a boyfriend?",
            "answer": "Yes"
        },
        {
            "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
            "question": "Based on the context, Did the plague affect people in Europe?",
            "answer": "yes, people in Italy, Europe",
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
            "answer": "unknown",
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "Based on the context, Is anti-matter made of electrons? ",
            "answer": "Unknown",
        },
    ]),
    pd.DataFrame([
         {
            "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
            "question": "Based on the context, Did the plague affect over 1,000 people?",
            "answer": "yes, 1,200 people",
        },
        {
            "context": "If inflation is occurring, leading to higher prices for basic necessities such as gas by 2 dollars. Do you think that inflation is good for society?",
            "question": "Based on the context, Is inflation good for society?",
            "answer": "Unknown"
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
            "answer": "unknown"
        }
    ]),
]

##############################################################################################################################


class ANLIR2Decomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = [' True', ' Neither', ' False']
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

    def _get_boost_decomp_examples(self, train_data, boost_id):
        seed = [69, 987][boost_id] 
        k_shot = 64
        random.seed(seed)
        np.random.seed(seed)

        data_train = pd.DataFrame(train_data)
        labels = [' False', ' True', ' Neither']
        num_per_class = int(np.ceil(k_shot / len(labels)))

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1
            if seed % 2 == 1:
                sub_df = data_train[data_train["targets_pretokenized"] == label].sample(num_per_class, random_state = seed)
            elif seed % 2 == 0:
                sub_df = data_train[data_train["targets_pretokenized"] != label].sample(num_per_class, random_state = seed)
            dfs.append(sub_df)
            total_in_context += num_per_class
            if total_in_context == k_shot:
                break

        booster_df = pd.concat(dfs).sample(frac=1, random_state=0)
        print(f"Selected: {len(booster_df)} in context examples.")
        return [
            booster_df
        ]

    def get_boost_decomp_examples(self, train_data, boost_id):
        if boost_id < 3:
            return [
                questioner_prompt_examples[boost_id],
                extraction_qa_examples[boost_id],
            ]
        else:
            icl_examples = self._get_boost_decomp_examples(train_data, boost_id-3)[0]
            return [
                icl_examples
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
        golds = []
        preds = []

        labels = set(test_data["targets_pretokenized"])
        labels = [l.lower().strip() for l in labels]

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            if ind in expt_log:
                pred = expt_log[ind]["pred"]
                gold = expt_log[ind]["gold"]
            else:
                icl_str = ""

                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        icl_str += f"{s_row['inputs_pretokenized']}{s_row['targets_pretokenized']}\n\n"

                text = row["inputs_pretokenized"]
                text = text.replace("True, False, or Neither?", "").strip().strip("\n")
                text = text + " True, False, or Neither? "
                gold = row["targets_pretokenized"]
                prompt = f"{icl_str}{{text:}}"
                pmp = prompt.format(text=text)
                if i == 0:
                    print(pmp)

                raw_answer = get_response(
                    pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=20,
                )

                answer = raw_answer.strip().lower()
                answer = answer.split("\n")
                answer = [a for a in answer if a]
                answer = [
                    a for a in answer if any(l.lower() in a.lower() for l in labels)
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
                is_maybe = "neither" in answer.split()
                pred = "Neither"
                if is_yes and (not is_maybe and not is_no):
                    pred = "True"
                if is_no and (not is_maybe and not is_yes):
                    pred = "False"
                if is_maybe and (not is_no and not is_yes):
                    pred = "Neither"

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

            golds.append(gold)
            preds.append(pred)

        report = classification_report(golds, preds, output_dict=True)
        return expt_log, report["accuracy"]

    def get_extraction(self, question, passage, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        if "Based on the context," in prompt_suffix:
            question_prefix = " Based on the context,"
        else:
            question_prefix = ""
        extract_prompt = f"{prompt_suffix}\n\nContext: {{passage:}}\nQuestion:{question_prefix} {question}\nAnswer:"
        extract_pmp = extract_prompt.format(passage=passage)
        answer = get_response(
            extract_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
        answer = answer.replace(",", "").replace(".", "").replace("?", "")
        answer = [a for a in answer.split("\n") if a]
        if answer:
            answer = answer[0]
        else:
            answer = passage
        return answer, extract_pmp

    def get_question(self, statement, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        question_prompt = f"{prompt_suffix}\n\nStatement: {{statement:}}\nQuestion:"
        question_pmp = question_prompt.format(statement=statement)
        answer = get_response(
            question_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
        answer = answer.replace("Question: ", "")
        answer = [a for a in answer.split("\n") if a]
        if answer:
            answer = answer[0].strip()
        # answer = ''
        statement = statement.strip().strip(".")
        if (
            not answer
            or statement.lower() == answer.lower()
            or not answer.strip().endswith("?")
        ):
            answer = f"{statement}. Yes, no, or unknown?"
        answer = answer.split("\n")[0]
        return answer, question_pmp

    def resolve_pred(self, answer):
        is_yes = "yes" in answer.split() or "true" in answer.split()
        is_no = "no" in answer.split() or "false" in answer.split()
        is_maybe = "maybe" in answer.split() or "maybe" in answer.split()

        pred = "Neither"
        if is_yes and (not is_maybe and not is_no):
            pred = "True"
        if is_no and (not is_maybe and not is_yes):
            pred = "False"
        if is_maybe and (not is_no and not is_yes):
            pred = "Neither"

        return pred

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest, run_limit=-1)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest, run_limit=1000)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train, indecisive_ans="neither")
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            report = classification_report(labels, [p[i] for p in all_boost_preds], output_dict=True)
            individual_accuracies.append(report["accuracy"])
            print(report)
            print("\n\n")
        report = classification_report(labels, preds, output_dict=True)
        print(report)
        return expt_log, expt_log_train, report["accuracy"], individual_accuracies    
    
    def _run_decomp_single_data(
        self, test_data, boost_dfs, manifest, overwrite_manifest, run_limit = -1
    ):
        expt_log = {}
        all_boost_preds = []
        labels = []

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            prompts_across_boost = []
            preds_across_boost = []

            if i == run_limit:
                break
            
            text = row["inputs_pretokenized"]
            gold = row["targets_pretokenized"].strip()
            passage = text.split("\n")[0]
            statement = (
                text.split("\n")[-1]
                .replace("True, False, or Neither?", "")
                .strip()
                .strip("\n")
                .replace("Question: ", "")
            )
            for boost_num, boost_examples in enumerate(boost_dfs):
                all_prompts = []

                # question / extract prompt
                if boost_num < 3:
                    question, question_final_prompt = self.get_question(
                        statement, questioner_prompt, boost_examples[0], manifest, overwrite_manifest
                    )
                    all_prompts.append(question_final_prompt)

                    open_answer_f, extraction_final_prompt = self.get_extraction(
                        question,
                        passage,
                        extraction_qa,
                        boost_examples[1],
                        manifest,
                        overwrite_manifest,
                    )
                    all_prompts.append(extraction_final_prompt)
                    if i == 0:
                        print("\n".join(all_prompts))
                    answer_f = open_answer_f.lower()
                    pred = self.resolve_pred(answer_f)
                    pred = pred.strip().lower()

                    preds_across_boost.append(pred)

                # just ICL
                elif boost_num >= 3:
                    icl_str = ""
                    for s_ind, s_row in boost_examples[0].iterrows():
                        if s_row["targets_pretokenized"].strip() == "True":
                            demo_label = "yes"
                        elif s_row["targets_pretokenized"].strip()  == "False":
                            demo_label = "no"
                        else:
                            demo_label = "unknown"

                        s_text = s_row["inputs_pretokenized"]
                        s_passage = s_text.split("\n")[0]
                        s_statement = (
                            s_text.split("\n")[-1]
                            .replace("True, False, or Neither?", "")
                            .strip()
                            .strip("\n")
                            .replace("Question: ", "")
                        )
                        icl = f"Statement: {s_statement}\nAnswer: {demo_label}"
                        icl_str += f"{icl}\n\n"

                    description = "Is the statement Yes, No, or Unknown?"
                    prompt = f"{description}\n\n{icl_str}Statement: {{statement:}}\nAnswer:"
                    pmp = prompt.format(statement=statement)
                    if i == 0:
                        print("PMP ICL")
                        print(pmp)
                    pred = get_response(
                        pmp,
                        manifest,
                        overwrite=bool(overwrite_manifest),
                        max_toks=10,
                        stop_token="\n",
                    )
                    pred = pred.lower().strip()
                    pred = pred.replace(".", "").replace(",", "").replace("Label: ", "").replace("Sentiment:", "")
                    pred = [p for p in pred.split("\n") if p]
                    if pred:
                        pred = pred[0]
                    else:
                        pred = ""

                    all_prompts.append(pmp)
                    prompts_across_boost.append(all_prompts)
                    pred = self.resolve_pred(pred).lower()
                    preds_across_boost.append(pred)
                gold = gold.strip().lower()

            expt_log[ind] = {
                "ind": ind,
                "preds_boost": preds_across_boost,
                "prompts": prompts_across_boost,
                "example": text,
                "pred": pred,
                "gold": gold,
            }
            all_boost_preds.append(preds_across_boost)
            labels.append(gold)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 5
    task_name = "anli_r2"
    data_dir = f"{DATA_DIR}/P3/data_feather/anli_GPT_3_style_r2"
    decomp = ANLIR2Decomp(task_name, data_dir, val_split="test")
    decomp.run(args)


if __name__ == "__main__":
    main()