#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from nltk.corpus import stopwords
from datasets import load_dataset

stops = set(stopwords.words("english"))

from sklearn.metrics import classification_report
from utils import get_response, InputOutputPrompt, load_hf_data

from decomposition import Decomposition, get_args

extract_relevant_phrase = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['passage']}",
    output_formatter=lambda x: f"Extract: {x['extract']}",
    required_keys=["passage", "extract"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Extract the phrase containing the pronoun.\n\n"
)
extract_relevant_phrase_examples = [
    pd.DataFrame([
        {
            "passage": "Jane's mom went to the shop to buy Jane a backpack for \"her\" first day of kindergarten.",
            "extract": "phrase containing \"her\": \"her\" first day"
        },
        {
            "passage": "The musicians performed in the park and the crowd loved \"them\". The crowd cheered for them.",
            "extract": "phrase containing \"them\": crowd loved \"them\""
        },
        {
            "passage": "Jeff gave his son some money because \"he\" wanted to buy lunch.",
            "extract": "phrase containing \"he\": \"he\" wanted to buy"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "The dog chased the cat. The cat ran up a tree and \"it\" waited at the top.",
            "extract": "phrase containing \"it\": \"it\" waited at the top"
        },
        {
            "passage": "The musicians performed in the park and the crowd loved \"them\". The crowd cheered for them.",
            "extract": "phrase containing \"them\": crowd loved \"them\""
        },
        {
            "passage": "John couldn't see the stage with Billy in front of him because \"he\" is so short.",
            "extract": "phrase containing \"he\": \"he\" is so short"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "The candle gave some light during the blackout, but after a while \"it\" also burned out.",
            "extract": "phrase containing \"it\": \"it\" also burned out"
        },
        {
            "passage": "Mark stocked the pantry with \"his\" son Jack's favorite cookies.",
            "extract": "phrase containing \"his\": \"his\" son Jack's"
        },
        {
            "passage": "Mary invited Jenna to \"her\" birthday party, but didn't invite Anna.",
            "extract": "phrase containing \"her\": \"her\" birthday party"
        }
    ]),
]

convert_reason_to_q = InputOutputPrompt(
    input_formatter=lambda x: f"Input: {x['input']}",
    output_formatter=lambda x: f"Question: {x['question']}",
    required_keys=["input", "question"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Rewrite the input as a question.\n\n"
)
convert_reason_to_q_examples = [
    pd.DataFrame([
        {
            "input": "it was made of glass",
            "question": "What was made of glass?"
        },
        {
            "input": "they are funny",
            "question": "Who or what are funny?"
        },
        {
            "input": "he drowned",
            "question": "Who drowned?"
        },
        {
            "input": "wrap around them",
            "question": "Wrap around who or what?"
        },
        {
            "input": "his cat is black",
            "question": "Whose cat is black?"
        },
        {
            "input": "laugh at them",
            "question": "Laugh at who?"
        },
        {
            "input": "her friend jennfier",
            "question": "Whose friend Jennifer?"
        }
    ]),
    pd.DataFrame([
        {
            "input": "it was made of glass",
            "question": "What was made of glass?"
        },
        {
            "input": "they are funny",
            "question": "Who or what are funny?"
        },
        {
            "input": "he drowned",
            "question": "Who drowned?"
        },
        {
            "input": "wrap around them",
            "question": "Wrap around who or what?"
        },
        {
            "input": "his cat is black",
            "question": "Whose cat is black?"
        },
        {
            "input": "laugh at them",
            "question": "Laugh at who?"
        },
        {
            "input": "her friend jennfier",
            "question": "Whose friend Jennifer?"
        }
    ]),
    pd.DataFrame([
        {
            "input": "it was made of glass",
            "question": "What was made of glass?"
        },
        {
            "input": "they are funny",
            "question": "Who or what are funny?"
        },
        {
            "input": "he drowned",
            "question": "Who drowned?"
        },
        {
            "input": "wrap around them",
            "question": "Wrap around who or what?"
        },
        {
            "input": "his cat is black",
            "question": "Whose cat is black?"
        },
        {
            "input": "laugh at them",
            "question": "Laugh at who?"
        },
        {
            "input": "her friend jennfier",
            "question": "Whose friend Jennifer?"
        }
    ]),
]

answer_q_in_passage = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['passage']}\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["passage", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Answer the question.\n\n"
)
answer_q_in_passage_examples = [
    pd.DataFrame([
        {
            "passage": "Jane's mom went to the shop to buy Jane a backpack for her first day of kindergarten.",
            "question": "Whose first day?",
            "answer": "Jane"
        },
        {
            "passage": "Mrs. Jenna told Fred she loved him.",
            "question": "Who loved him?",
            "answer": "Mrs. Jenna"
        },
        {
            "passage": "Joe gave Mark some money so he could buy lunch.",
            "question": "Who could buy lunch?",
            "answer": "Mark"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Joe gave Mark some money so he could buy lunch.",
            "question": "Who could buy lunch?",
            "answer": "Mark"
        },
        {
            "passage": "Jane's mom went to the shop to buy Jane a backpack for her first day of kindergarten.",
            "question": "Whose first day?",
            "answer": "Jane"
        },
        {
            "passage": "Mark stocked the pantry with his son Jack's favorite cookies.",
            "question": "Whose son?",
            "answer": "Mark"
        },
    ]),
    pd.DataFrame([
        {
            "passage": "The candle burned out after some time. It dripped a lot of wax.",
            "question": "What dripped?",
            "answer": "The candle"
        },
        {
            "passage": "Mark stocked the pantry with his son Jack's favorite cookies.",
            "question": "Whose son?",
            "answer": "Mark"
        },
        {
            "passage": "Mary invited Jenna to her birthday party.",
            "question": "Whose birthday party?",
            "answer": "Mary"
        }
    ]),
]

class WSCDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            extract_relevant_phrase_examples[boost_id],
            convert_reason_to_q_examples[boost_id],
            answer_q_in_passage_examples[boost_id],
        ]

    def read_data(self, save_dir, overwrite_data):
        return load_hf_data(save_dir, self.task_name, self.val_split, "SetFit/wsc_fixed", overwrite_data)

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
            gold = row['label_text']
            if gold == 'False':
                gold = 'No'
            elif gold == 'True':
                gold = 'Yes'
            pronoun = row['span2_text']
            subject = row['span1_text']
            text_toks = text.split(" ")
            text_toks_prefix = text_toks[:row['span2_index']]
            text_toks_suffix = text_toks[row['span2_index']+len(pronoun.split()):]
            text_toks = text_toks_prefix + [f'"{pronoun}"'] + text_toks_suffix
            passage = " ".join(text_toks).strip(".").strip() + "."
            question = f"Question: In the passage above, does the pronoun \"{pronoun}\" refer to {subject}?"

            icl_str = ""
            if do_few_shot:
                for s_ind, s_row in few_shot_df.iterrows():
                    s_text = s_row['text']
                    s_gold = s_row['label_text']
                    if s_gold == 'False':
                        s_gold = 'No'
                    elif s_gold == 'True':
                        s_gold = 'Yes'
                    s_pronoun = s_row['span2_text']
                    s_subject = s_row['span1_text']
                    s_text_toks = s_text.split(" ")
                    s_text_toks_prefix = s_text_toks[:s_row['span2_index']]
                    s_text_toks_suffix = s_text_toks[s_row['span2_index']+len(s_pronoun.split()):]
                    s_text_toks = s_text_toks_prefix + [f'"{s_pronoun}"'] + s_text_toks_suffix
                    s_passage = " ".join(s_text_toks).strip(".").strip() + "."
                    s_question = f"Passage: {s_passage}\nQuestion: In the passage above, does the pronoun \"{s_pronoun}\" refer to {s_subject}?"
                    icl_str += f"{s_question}\nAnswer: {s_gold}\n\n"

            prompt = f"{icl_str}Passage: {{passage:}}\n{question}\nAnswer:"
            pmp = prompt.format(passage=passage)
            if i == 0:
                print(pmp)
            raw_answer = get_response(
                pmp,
                manifest,
                overwrite=bool(overwrite_manifest),
                max_toks=30,
            )
            answer = raw_answer.split("\n")
            answer = [a for a in answer if a]
            if len(answer) <= 0:
                answer = ""
            else:
                answer = answer[0]
            answer = " ".join(
                [a.strip(",").strip(".").strip() for a in answer.split()]
            )

            is_yes = "yes" in answer.lower().split()
            is_no = "no" in answer.lower().split()
            if is_yes and not is_no:
                pred = "Yes"
            elif is_no and not is_yes:
                pred = "No"
            else:
                pred = ""

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

    def question_answer(
        self,
        all_prompts,
        boost_exs,
        passage,
        original_passage,
        pronoun,
        manifest,
        overwrite_manifest,
    ):
        prompt_suffix = all_prompts[0](boost_exs[0])
        extract_prompt = (
            f"{prompt_suffix}\n\nPassage: {{passage:}}\nExtract: phrase containing \"{{pronoun:}}\": "
        )
        extract_pmp = extract_prompt.format(passage=passage, pronoun=pronoun)
        relevant_phrase = get_response(
            extract_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
        relevant_phrase = relevant_phrase.split("\n")[0]
        relevant_phrase = relevant_phrase.replace('"', '')
        prompt_suffix = all_prompts[1](boost_exs[1])
        convert_prompt = f"{prompt_suffix}\n\nInput: {{relevant_phrase:}}\nQuestion:"
        convert_pmp = convert_prompt.format(relevant_phrase=relevant_phrase)
        converted = get_response(
            convert_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
        converted = converted.split("\n")[0].replace("Question: ", "")
        prompt_suffix = all_prompts[2](boost_exs[2])
        answer_prompt = f"{prompt_suffix}\n\nPassage: {{passage:}}\nQuestion: {{converted:}}\nAnswer:"
        answer_pmp = answer_prompt.format(passage=original_passage, converted=converted)
        answer = get_response(
            answer_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
        answer = answer.split("\n")[0].strip("'s").strip().replace("Answer: ", "").replace("A: ", "").strip()
        return answer, extract_pmp, convert_pmp, answer_pmp

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
            if i == run_limit:
                break
            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                all_prompts = []
                text = row['text']
                gold_answer = row['label_text']
                pronoun = row['span2_text']
                gold = row['span1_text']
                text_toks = text.split(" ")
                text_toks_prefix = text_toks[:row['span2_index']]
                text_toks_suffix = text_toks[row['span2_index']+len(pronoun.split()):]
                text_toks = text_toks_prefix + [f'\"{pronoun}\"'] + text_toks_suffix
                passage = " ".join(text_toks).strip(".").strip() + "."
                original_passage = text.strip(".").strip() + "."

                # gold = question.split("refer to")[-1].replace("?", "").strip().lower()
                gold_split = gold.split()
                if gold_split[0] in stops:
                    gold = " ".join(gold_split[1:])
                (
                    qa_answer,
                    extract_prompt,
                    convert_prompt,
                    answer_prompt,
                ) = self.question_answer(
                    [extract_relevant_phrase, convert_reason_to_q, answer_q_in_passage],
                    boost_examples,
                    passage,
                    original_passage,
                    pronoun,
                    manifest,
                    overwrite_manifest,
                )
                all_prompts.append(extract_prompt)
                all_prompts.append(convert_prompt)
                all_prompts.append(answer_prompt)

                if i == 0:
                    print(extract_prompt)
                    print(convert_prompt)
                    print(answer_prompt)

                answer_no_stop = " ".join(
                    [a for a in qa_answer.lower().split() if a not in stops]
                ).lower()
                gold_no_stop = " ".join([a for a in gold.lower().split() if a not in stops]).lower()
                answer_no_stop = answer_no_stop.strip("s")
                gold_no_stop = gold_no_stop.strip("s")
                if (
                    answer_no_stop.strip() == gold_no_stop.strip()
                    or gold_no_stop.strip() == answer_no_stop.strip()
                ):
                    pred = "True"
                else:
                    pred = "False"
                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)
            
            entry = {
                "ind": ind,
                "example": text,
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "gold": gold_answer,
            }
            expt_log[ind] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(gold_answer)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    task_name = "super_glue_wsc"
    data_dir = "SetFit/wsc_fixed"
    decomp = WSCDecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
