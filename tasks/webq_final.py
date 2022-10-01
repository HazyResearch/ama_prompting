#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import ast
from tqdm.auto import tqdm
from decomposition import Decomposition, get_args
from utils import get_response, InputOutputPrompt, accuracy_span_overlap, load_hf_data

extract = InputOutputPrompt(
    input_formatter=lambda x: f"Question: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Produce distinct questions.\n\n"
)

more_info_examples = [
    pd.DataFrame([
        {
            "question": "who plays Carrie Bradshaw in sex and the city?",
            "answer": "Caroline \"Carrie\" Bradshaw is a fictional character from the HBO franchise Sex and the City, portrayed by Sarah Jessica Parker."
        },
        {
            "question": "what are the elements in air?",
            "answer": "By mole fraction (i.e., by number of molecules), dry air contains 78.08% nitrogen, 20.95% oxygen, 0.93% argon, 0.04% carbon dioxide, and small amounts of other gases"
        },
        {
            "question": "what is HP company?",
            "answer": "HP Inc. is an American multinational information technology company headquartered in Palo Alto, California, that develops personal computers (PCs)"
        },
        {
            "question": "when was the last season of FRIENDS released?",
            "answer": "The series finale aired on May 6, 2004, and was watched by around 52.5 million American viewers, making it the fifth-most-watched series finale in television history"
        }
    ]),

]

answer = InputOutputPrompt(
    input_formatter=lambda x: f"Context: {x['context']}\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["context", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Answer the question.\n\n"
)

answer_question = [
    pd.DataFrame([
        {
            'context': 'The nearest airport to Palm Springs is Indio/Palm Springs (PSP) Airport which is 2.1 miles away. ', 
            'question': 'what airport is closest to palm springs?', 
            'answer': 'Palm Springs International Airport'
        }, 
        {
            'context': 'Martin Luther King earned his Bachelor of Divinity degree from Crozer Theological Seminary, followed by a doctorate in Systematic Theology from Boston University.', 
            'question': 'what degree did martin luther king get?', 
            'answer': 'Bachelor of Divinity'
        }, 
        {
            'context': 'The Niger river runs in a crescent through Libya, Mali, Niger, on the border with Benin and then through Nigeria.', 
            'question': 'what countries does the niger river flow through?', 
            'answer': 'Libya'
        }, 
        {
            'context': 'Puerto Rico is a territory of the United States and uses the U.S. dollar. ', 
            'question': 'what type of currency is used in puerto rico?', 
            'answer': 'United States dollar'
        }, 
        {
            'context': 'kitt was voice most often by William daniels.', 
            'question': 'who played kitt in knight rider?', 
            'answer': 'William Daniels'
        }
    ]),
    pd.DataFrame([
        {
            'context': 'leonardo da vinci invented the parachute, the helicopter, double hull, an armored fighting vehicle,', 
            'question': 'what inventions did leonardo da vinci made?', 
            'answer': 'Double hull'
        }, 
        {
            'context': "The French franc (F) was the national currency of France prior to France's adoption of the euro (EUR) in January 2002.", 
            'question': 'what currency is used in france before euro?', 
            'answer': 'French franc'
        },
        {
            'context': 'The Isthmus of Panama, contains the country of Panama and the panama canal.', 
            'question': 'where is isthmus of panama located?', 
            'answer': 'Costa Rica'
        }, 
        {
            'context': 'Hurricane Irene was a large and destructive tropical cyclone which affected much of the Caribbean and East Coast', 
            'question': 'where did hurricane irene?', 
            'answer': 'Eastern United States'
        }, 
        {
            'context': 'Rihanna acted in This is the End and Battleship.', 
            'question': 'what movie did rihanna play in?', 
            'answer': 'This Is the End'
        }
    ]),
    pd.DataFrame([
        {
            'context': 'st vincent de paul is buried in the 6th arrondisment of Paris.', 
            'question': 'where is st vincent de paul buried?', 
            'answer': 'Paris'
        }, 
        {
            'context': 'Thomas Luther "Luke" Bryan (born July 17, 1976) is an American country singer and songwriter from Leesburg.', 
            'question': 'where is luke bryan from?', 
            'answer': 'Leesburg'
        }, 
        {
            'context': "Klum and Seal got married on 10 May 2005 on a beach in Mexico near Seal's home on Costa Careyes. ", 
            'question': 'where did heidi klum and seal get married?', 
            'answer': 'Mexico'}, 
        {
            'context': 'Tarantino starred in pulp fiction, grindhouse and others.', 
            'question': 'what movies did quentin tarantino star in?', 
            'answer': 'Grindhouse'
        }, 
        {
            'context': 'Countries that are sometimes considered to be entirely or partially part of the Balkans are Croatia, Serbia, Lake Prespa.', 
            'question': 'what country is located in the balkan peninsula?', 
            'answer': 'Lake Prespa'
        }
    ])

]

prefix_select_zeroshot = """Answer the question.\n\n"""


class WebQDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def read_data(self, save_dir, overwrite_data):
        return load_hf_data(save_dir, self.task_name, self.val_split, "web_questions", overwrite_data)

    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            more_info_examples[0],
            answer_question[boost_id],
        ]

    def zero_few_baseline(
        self,
        test_data,
        few_shot_df,
        manifest,
        overwrite_manifest,
        prompt_suffix="",
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
                question = row.question
                if isinstance(row.answers, str):
                    label = ast.literal_eval(row.answers)
                else:
                    label = row.answers.tolist()
                gold = label

                icl_str = ""
                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        s_question = s_row.question
                        if isinstance(s_row.answers, str):
                            label = ast.literal_eval(s_row.answers)
                        else:
                            label = s_row.answers.tolist()
                        icl_str += f"Question: {s_question}\nAnswer: {label[0]}\n\n"

                prompt = (
                    icl_str
                    + "Question: {question:}"
                    + prompt_suffix
                    + "\nAnswer:"
                )

                if i == 0:
                    print(prompt.format(question=question))
                prompt = prompt.format(question=question)
                raw_answer = get_response(
                    prompt, #prompt.format(question=question),
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=20,
                    stop_token="\n\n",
                )
                pred = raw_answer.strip("\n").strip().lower()
                entry = {
                    "ind": ind,
                    "example": question,
                    "base_prompt": prompt,
                    "raw_answer": raw_answer,
                    "pred": pred,
                    "gold": gold,
                }
                expt_log[ind] = entry

            preds.append([pred])
            labels.append(gold)
        metric = accuracy_span_overlap(preds=preds, golds=labels)
        return expt_log, metric

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest, run_limit=1000)
        
        # Do WS
        boost_test, boost_train = [], []
        for p in all_boost_preds:
            samples = [lf[1] for lf in p]
            boost_test.append(samples)
        for p in all_boost_train_preds:
            samples = [lf[1] for lf in p]
            boost_train.append(samples)

        preds = self.merge_boosted_preds(boost_test, boost_train, train_labels, expt_log, expt_log_train)
        preds = [(x,y) for x,y in zip([p[0][0] for p in all_boost_preds], preds)]

        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            individual_accuracies.append(accuracy_span_overlap(preds=[p[i] for p in all_boost_preds], golds=labels))

        metric = accuracy_span_overlap(preds=preds, golds=labels)
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

            question = row.question
            if isinstance(row.answers, str):
                label = ast.literal_eval(row.answers)
            else:
                label = row.answers.tolist()

            gold = label
            prompts_across_boost = []
            preds_across_boost = []

            # extract context
            prompt_suffix = extract(boost_dfs[0][0])
            prompt = (
                    prompt_suffix + "\n\Question: {question:}\nAnswer:"
            )
            more_info_answer = get_response(
                    prompt.format(question=question),
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=20,
                    stop_token="\n\n",
            )


            for boost_examples in boost_dfs:
                all_prompts = []
                prompt_suffix = answer(boost_examples[1])
                prompt = (
                    prompt_suffix + "\n\nContext: {text:}\nQuestion: {question:}\nAnswer:"
                )
                if i == 0:
                    print(prompt.format(text=more_info_answer, question=question))

                all_prompts.append(prompt.format(text=more_info_answer, question=question))
                raw_answer = get_response(
                    prompt.format(text=more_info_answer, question=question),
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=20,
                    stop_token="\n\n",
                )
                pred = raw_answer.split("\n")[0].strip().lower()
                prompts_across_boost.append(all_prompts)
                preds_across_boost.append((more_info_answer, pred))
            
            entry = {
                "ind": ind,
                "example": question,
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
    task_name = "webq"
    data_dir = "webq"
    webq = WebQDecomp(task_name, data_dir, val_split="test")
    webq.run(args)


if __name__ == "__main__":
    main()
