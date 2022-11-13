#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
import pandas as pd
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

questioner = InputOutputPrompt(
    input_formatter=lambda x: f"Statement: {x['statement']}",
    output_formatter=lambda x: f"Question: {x['question']}",
    input_output_sep="\n",
    example_sep="\n\n",
    required_keys=["question", "statement"],
    instruction="Rewrite the statement as a question.\n\n"
)

questioner_examples = [
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test hard?"
        },
        {
            "statement": "it was a good idea to buy your parents gifts",
            "question": "Was it a good idea to buy your parents gifts?"
        },
        {
            "statement": "The 20 cans will arrive in the grocery store tomorrow.",
            "question": "Will the 20 cans arrive in the grocery store tomorrow?"
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?"
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?"
        }
    ]),

    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not",
            "question": "Was the test hard?"
        },
        {
            "statement": "it was a good idea to buy your parents gifts",
            "question": "Was it a good idea to buy your parents gifts?"
        },
        {
            "statement": "The 20 cans will arrive in the grocery store tomorrow.",
            "question": "Will the 20 cans arrive in the grocery store tomorrow?"
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?"
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?"
        }
    ]),

    pd.DataFrame([
        {
            "statement": "tennis can be played on grass courts",
            "question": "Can tennis be played on grass courts?",
        },
        {
            "statement": "the artist painted a picture of the apple in a bowl.",
            "question": "Did the artist paint a picture of an apple in a bowl?",
        },
        {
            "statement": "mary is unhappy with tim.",
            "question": "Is mary unhappy with Tim?",
        },
        {
            "statement": "after school, Jim was going to go to the park",
            "question": "Was Jim going to go to the park after school?",
        },
    ]),

    pd.DataFrame([
        {
            "statement": "she prefers kittens over puppies",
            "question": "What does she prefer over puppies?\nAnswer: kittens",
        },
        {
            "statement": "Max and his wife went on a trip to Europe",
            "question": "Where did Max and his wife go on a trip?\nAnswer: Europe",
        },
        {
            "statement": "jared was born during the war in 1942",
            "question": "Jared was born during a war in which year?\nAnswer: 1942",
        },
        {
            "statement": "it took jenna 7 attempts to solve the problem",
            "question": "How many attempts did it take Jenna to solve the problem?\nAnswer: 7",
        },
    ]),
]

openended_qa = InputOutputPrompt(
    input_formatter=lambda x: f"Context: {x['passage']}\n\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    input_output_sep="\n\n",
    example_sep="\n\n----\n\n",
    required_keys=["question", "statement", 'answer'],
    instruction="Answer the question. If there is no evidence in the context, return \"Unknown\".\n\n"
)

openended_qa_examples = [
    pd.DataFrame([
        {
            "passage": "Jenna's 10th birthday was yesterday evening and at least 10 of her friends attended the party.",
            "question": "Did 10 friends attend Jenna's party?",
            "answer": "Unknown, at least 10"
        },
        {
            "passage": "The bullies attacked John when he was walking through the elementary school parking lot and then got sent to the teacher's office.",
            "question": "Did the bullies attack John in the teacher's office?",
            "answer": "No, parking lot"
        },
        {
            "passage": "WISS discovered a new monkey disease in a remote tribe in the Amazon rainforrest last week. It was highly contagious.",
            "question": "Did WISS discover a new disease?",
            "answer": "Yes, new monkey disease"
        },
    ]),

    pd.DataFrame([
        {
            "passage": "Jenna's birthday was yesterday evening and at least 10 of her friends attended the party.",
            "question": "Did 10 friends attend Jenna's party?",
            "answer": "unknown, at least 10"
        },
        {
            "passage": "The bullies punched John when he was walking through the elementary school parking lot. They punched 3 times.",
            "question": "Did the bullies punch John 4 time?",
            "answer": "No, 3 times"
        },
        {
            "passage": "WISS discovered a new monkey disease in a remote tribe in the Amazon rainforrest last week. It was highly contagious.",
            "question": "Did WISS discover a new species of monkeys?",
            "answer": "Unknown"
        },
    ]),

    pd.DataFrame([
        {
            "passage": "The doctor performed surgery at the hospital and then went to the school to pick up her son.",
            "question": "Was the surgery successful?",
            "answer": "Unknown"
        },
        {
            "passage": "As soon as the book was released, it became a New York Times fiction bestseller.",
            "question": "Is the book non-fiction?",
            "answer": "No, Fiction bestseller"
        },
        {
            "passage": "During the presidential election polls last week, Jeff had 15% more votes than John",
            "question": "Were Jack and John running for president?",
            "answer": "Yes, presidential election"
        },
    ]),

    pd.DataFrame([
        {
            "passage": "According to Biraben, the plague was present somewhere in Italy in every year between 1346 and 1671",
            "question": "Where was the plague present?",
            "answer": "somewhere in Italy"
        },
        {
            "passage": "Jenna's birthday was yesterday evening and at least 10 of her friends attended the party.",
            "question": "How many of Jenna's friends attended?",
            "answer": "at least 10"
        },
        {
            "passage": "Mitsubishi Motor Corp's vehicle sales fell by 42 percent in June",
            "question": "When did Mitsubishi's sales fall?",
            "answer": "June"
        },
        {
            "passage": "The bullies attacked in the elementary school parking lot and then got sent to the teacher's office.",
            "question": "Who or what did the bullies punch?",
            "answer": "Unknown"
        },
    ]),
]


cloze_convertor = InputOutputPrompt(
    input_formatter=lambda x: f"Example: {x['passage']}",
    output_formatter=lambda x: f"Output: {x['question']}",
    input_output_sep="\n",
    example_sep="\n\n",
    required_keys=["question", "passage"],
    instruction=""
)
cloze_examples = [
    pd.DataFrame([
        {
            "passage": "Barrack Obama believes the best novel is Harry Potter.",
            "question": "Barrack Obama believes the best novel is Harry",
        },
        {
            "passage": "The girl invited 12 friends to her birthday party last week.",
            "question": "The girl invited 12 friends to her birthday ",
        },
        {
            "passage": "Apple computers are worse than Dell computers.",
            "question": "Apple computers are worse",
        },
        {
            "passage": "Welcome to New York.",
            "question": "Welcome to New"
        }
    ]),
]

cloze_choices = InputOutputPrompt(
    input_formatter=lambda x: f"Example: {x['example']}\nList alternatives:\n- {x['alternatives1']}\n- {x['alternatives2']}\n- {x['alternatives3']}",
    output_formatter=lambda x: f"",
    input_output_sep="",
    example_sep="\n\n",
    required_keys=["example", "alternatives1", "alternatives2", "alternatives3"],
    instruction="Output a list of unique alternatives for each example.\n\n"
)

cloze_choice_examples = [
    pd.DataFrame([
        {
            "example": "Barrack Obama believes the",
            "alternatives1": "best novel is Harry Potter",
            "alternatives2": "worst book is Harry Potter",
            "alternatives3": "United States is great"
        },
        {
            "example":"The Beatles were honored in:",
            "alternatives1":"Buckingham Palace",
            "alternatives2":"Mexico",
            "alternatives3":"Tower of London"
        },
        {
            "example":"Jerry Baker:",
            "alternatives1":"is part of a soccer team",
            "alternatives2":"is not part of a soccer team",
            "alternatives3":"is a character in a book"
        },
    ])
]

cloze_completion = InputOutputPrompt(
    input_formatter=lambda x: f"Select One Choice:\n1. {x['alternatives1']}\n2. {x['alternatives2']}\n3. {x['alternatives3']}\n\nPassage: {x['passage']}\n\nThe passage \"Passage\" states: {x['statement']}: \"Choice\":",
    output_formatter=lambda x: f"{x['answer']}",
    input_output_sep=" ",
    example_sep="\n\n----\n\n",
    required_keys=["passage", "alternatives1", "alternatives2", "alternatives3", "statement", "answer"],
    instruction="Select one choice from the passage.\n\n"
)

cloze_completion_examples = [
    pd.DataFrame([
        {
            "passage": "Microsoft Corporation produces computer software, consumer electronics, and personal computers. It is headquartered at the Microsoft Redmond campus located in Redmond, Washington, United States.",
            "alternatives1": "consumer electronics",
            "alternatives2": "Play Stations",
            "alternatives3": "cameras",
            "statement": "Microsoft Corporation sells",
            "answer": "consumer electronics"
        },
        {
            "passage":"Sir Elton Hercules John CH CBE is a British singer, pianist and reknowned composer. His nickname is the Rocket Man.",
            "alternatives1":"and tall man",
            "alternatives2":"and trombone player",
            "alternatives3":"and reknowned composer",
            "statement": "Sir Elton John is a musician",
            "answer": "and reknowned composer"
        },
        {
            "passage":"The Mac versus Windows PC debate has been going on for a long time.  Most people say the Windows PC three is superior. It comes down to personal preference.",
            "alternatives1":"Lenovo computers",
            "alternatives2":"Windows PC three",
            "alternatives3":"Dell computers",
            "statement": "Apple computers are superior to",
            "answer": " Windows PC three"
        },
    ])
]


class RTEDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, data_train, boost_id):
        if boost_id < 4:
            return [
                questioner_examples[boost_id],
                openended_qa_examples[boost_id],
            ]
        else:
            return [
                cloze_examples[0],
                cloze_choice_examples[0],
                cloze_completion_examples[0]
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
                if is_yes and (not is_no):
                    pred = "True"
                if is_no and (not is_yes):
                    pred = "False"
                elif not is_no and not is_yes:
                    pred = "False"
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

    def get_question(self, statement, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        quesiton_prompt = f"{prompt_suffix}\n\nStatement: {{statement:}}\nQuestion:"
        quesiton_prompt = quesiton_prompt.format(statement=statement).replace("\n\nAnswer:", "\nAnswer:")
        chopped_answer = get_response(
            quesiton_prompt,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50)
        chopped_answer = chopped_answer.split("\n")
        question = [ch for ch in chopped_answer if ch][0]
        answer = [ch for ch in chopped_answer if ch.startswith("Answer: ")]
        if answer:
            answer = answer[0].replace(",", "").replace(".", "").replace("?", "").replace("Answer: ", "")
            answer = " ".join([a for a in answer.split() if a not in stops])
        else:
            answer = ''
        
        if "A:" in question:
            statement = statement.strip(".")
            return f"{statement}. Yes or no?"
        return question, answer, quesiton_prompt

    def open_qa(self, question, passage, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        qa_prompt = f"{prompt_suffix}\n\n----\n\nContext: {{passage:}}\n\nQuestion: {{question:}}\n\nAnswer:"
        qa_prompt = qa_prompt.format(passage=passage, question=question)
        answer = get_response(
            qa_prompt, 
            manifest, 
            overwrite=bool(overwrite_manifest),
            max_toks=50
        )
        answer = answer.replace(",", "").replace(".", "").replace("?", "")
        answer = [a for a in answer.split("\n") if a]
        if answer:
            answer = answer[0]
        else:
            answer = passage
        return answer, qa_prompt

    def resolve_pred(self, answer, open_answer):
        answer = answer.lower()
        is_yes = "yes" in answer.split() or "true" in answer.split()
        is_no = "no" in answer.split()  or "false" in answer.split()
        is_maybe = False
        answer = answer.replace("-", "")
        
        pred = "False" 
        if is_yes and (not is_maybe and not is_no) or (answer in open_answer or open_answer in answer):
            pred = "True"
        if is_no and (not is_maybe and not is_yes):
            pred = "False"
        return pred

    def get_choices_answer(self, chopped_answer, cuttoff, prompt, boost_ex, manifest, overwrite_manifest, get_choices_prompt=''):
        prompt_suffix = prompt(boost_ex)
        prompt = f"{prompt_suffix}\n\nExample: {{example:}}\nList alternatives:\n- {{cuttoff:}}\n"
        choices_answer = get_response(
            prompt.format(example=chopped_answer, cuttoff=cuttoff), 
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks = 30
        )
        choices_answer = choices_answer.split("\n\n")[0]
        choices_answer = choices_answer.split("\n")
        choices_answer = [a.replace("- ", "").strip() for a in choices_answer]
        choices_answer = [a for a in choices_answer if cuttoff.lower() not in a.lower()] 
        choices_answer = list(sorted(set(choices_answer)))
        choices_answer = choices_answer[:min(len(choices_answer), 2)]
            
        choices_answer = list(sorted(set(choices_answer)))
        choices_answer.append(cuttoff)
        choices_answer = [ch.strip(".") for ch in choices_answer]
        return choices_answer, prompt
    
    def get_chopping(self, question, prompt, boost_ex, manifest, overwrite_manifest, cuttoff_size=2, chopper_prompt=''):
        prompt_suffix = prompt(boost_ex)
        prompt = f"{prompt_suffix}\n\nExample: {{question:}}\nOutput:"
        chopped_answer = get_response(
            prompt.format(question=question), 
            manifest, 
            overwrite=bool(overwrite_manifest),
            max_toks = len(question.split())*4
        )
        chopped_answer = chopped_answer.split("\n")[0]  
        chopped_list = chopped_answer.split()
        question = question.split()
        cuttoff = [t for t in question if t not in chopped_list]
            
        cuttoff_str = " ".join(cuttoff).strip(".")
        chopped_list_str = " ".join(chopped_list).strip(".")
        if not cuttoff or chopped_list_str.endswith(cuttoff_str):
            chopped_list = question[0:-cuttoff_size]
            cuttoff = question[-cuttoff_size:]
        cuttoff = " ".join(cuttoff)
        chopped_answer = " ".join(chopped_list)
        cuttoff = cuttoff.strip(".")
        return chopped_answer, cuttoff, prompt

    def get_final_selection(self, choices_answer, passage, chopped_answer, prompt, boost_ex, manifest, overwrite_manifest, selector_prompt=''):
        prompt_suffix = prompt(boost_ex)
        select_choice_str = ""
        gold_choice = choices_answer[-1]
        other_choices = choices_answer[:-1]
        for num, ch in enumerate(choices_answer):
            select_choice_str += f"\n{num+1}. {ch}"
        prompt = f"{prompt_suffix}\n\n----\n\nSelect one Choice:{{choices_str:}}\n\nPassage: {{passage:}}\nThe passage \"Passage\" states: {{chopped_answer:}} \"Choice\": "
        
        select_answer = get_response(
            prompt.format(choices_str=select_choice_str, passage=passage, chopped_answer=chopped_answer), 
            manifest, 
            overwrite=bool(overwrite_manifest),
            max_toks = max(len(c.split()) for c in choices_answer)
        )
        select_answer = select_answer.lower()
        select_answer = select_answer.split("\n")[0].strip(".")
        
        if select_answer.lower() in gold_choice.lower():
            answer = "True"
        else:
            answer = "False"
        return answer, prompt

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
            input = row['inputs_pretokenized']
            gold = row['targets_pretokenized']
            passage = input.split("Question: ")[0].strip("\n")
            statement = input.split("Question: ")[-1].replace("True or False?", "")

            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            for boost_num, boost_examples in enumerate(boost_dfs):
                all_prompts = []

                if boost_num < 4:
                    question, proposed_answer, question_final_prompt = self.get_question(
                        statement, questioner, boost_examples[0], manifest, overwrite_manifest
                    )
                    if i == 0:
                        print("PROMPT:")
                        print(question_final_prompt)

                    open_answer, answer_final_prompt = self.open_qa(
                        question, passage, openended_qa, boost_examples[1], manifest, overwrite_manifest
                    )
                    if i == 0:
                        print("\nPROMPT:")
                        print(answer_final_prompt)
                    all_prompts.append(question_final_prompt)
                    all_prompts.append(answer_final_prompt)

                    open_answer = open_answer.replace("-", "")
                    open_answer = " ".join([a for a in open_answer.split() if a not in stops])
                    if proposed_answer:
                        answer = proposed_answer.replace("-", "")
                        answer = " ".join([a for a in answer.split() if a not in stops])
                        if all(wd in open_answer.lower() for wd in answer.lower().split()) or all(wd in answer.lower() for wd in open_answer.lower().split()):
                            pred = "True"
                        else:
                            pred = 'False'
                        if not answer.strip():
                            pred = 'False'
                    else:
                        pred = self.resolve_pred(open_answer.lower(), open_answer)
                else:
                    chopped_answer, cuttoff, chopper_prompt = self.get_chopping(
                        statement, cloze_convertor, boost_examples[0], manifest, overwrite_manifest, cuttoff_size=2)

                    choices_answer, choices_prompt = self.get_choices_answer(
                        chopped_answer, cuttoff, cloze_choices, boost_examples[1], manifest, overwrite_manifest)

                    pred, selector_prompt = self.get_final_selection(
                        choices_answer, passage, chopped_answer, cloze_completion, boost_examples[2], manifest, overwrite_manifest)

                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)

            entry = {
                "ind": ind,
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "example": input,
                "gold": gold,
            }
            expt_log[ind] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(gold)
        return expt_log, all_boost_preds, labels

def main():
    args = get_args()
    args.num_boost = 5
    task_name = "super_glue_rte"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_rte_GPT_3_style/"
    decomp = RTEDecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
