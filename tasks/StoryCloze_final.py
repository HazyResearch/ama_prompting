#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
import pandas as pd
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

questioner = InputOutputPrompt(
    input_formatter=lambda x: f"Statement: {x['statement']}",
    output_formatter=lambda x: f"Question: {x['question']}",
    required_keys=["statement", "question"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Rewrite the statement as a yes/no question.\n\n"
)

questioner_examples = [
    pd.DataFrame([
        {
            "statement": "Jonathan Samuels was born in the 70's.",
            "question": "Was Jonathan Samuels born in the 70's?"
        },
        {
            "statement": "Jerry bullied him and called him names",
            "question": "Did Jerry bully him and call him names?",
        },
        {
            "statement": "Sam and jade were going to go to the movies",
            "question": "Did did Sam and jade go to the movies?",
        },
        {
            "statement": "Chocolate is tasty, when I am feeling hungry.",
            "question": "Does chocolate taste good when you are hungry?",
        },
        {
            "statement": "Mark ran fast.",
            "question": "Did mark run fast?",
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
            "statement": "the test was not",
            "question": "Was the test hard?"
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?"
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?"
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?"
        }
    ])
]

openended_qa = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['context']}\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["passage", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction=""
)

openended_qa_examples = [
    pd.DataFrame([
        {
            "context": "My aunt is a nurse and she often talks about long hours at work. Last week was especially bad and she was constantly working many hours.",
            "question": "Was her work easy?",
            "answer": "No, it was hard work."
        },
        {
            "context": "My roommate was sick. She stayed home from work and school. She slept all day long and by the end of the day, she was feeling better.",
            "question": "Did the rest help her?",
            "answer": "Yes, she slept and felt better."
        },
        {
            "context": "Andy had always wanted a big kids bike. When he turned six Year's old he asked for a bike for his birthday. He did not know how to ride a bike. On Andy's birthday his mother gave him a bike.",
            "question": "Did he cry all night?",
            "answer": "No, Andy was happy because he got a bike."
        },
    ]),
    pd.DataFrame([
        {
            "context": "Anna's mother always told her to be confident even if she feels nervous on the inside",
            "question": "Does Anna always feel nervous on the inside?",
            "answer": "Unknown"
        },
        {
            "context": "Max and Jeff were extremely competitive at soccer, but Max was a lot better.",
            "question": "Was Jeff better than Max at soccer?",
            "answer": "No, Max was a lot better"
        },
        {
            "context": "When Judy and Jack went to school, they got in trouble with their teacher for being late. I didn't think it was very fair.",
            "question": "Did she think it was fair?",
            "answer": "No, she didn't think it was very fair."
        },
        {
            "context": "The FSP conference took place last week in Spain and representatives from 21 countries attended.",
            "question": "Did representatives from more than 20 countries attend FSP?",
            "answer": "Yes"
        },
    ]),
    pd.DataFrame([
        {
            "context": "My roommate was sick. She stayed home from work and school. She slept all day long and by the end of the day, she was feeling better.",
            "question": "Did the rest help her?",
            "answer": "Yes, she slept and felt better."
        },
        {
            "context": "It was a beautiful day outside. Bob decided to go for a walkk. Bob walked along the path and admired the scenery. He found a twenty dollar bill on the ground.",
            "question": "Was he disappointed?",
            "answer": "No, he was happy he got money."
        },
        {
            "context": "My aunt is a nurse and she often talks about long hours at work. Last week was especially bad and she was constantly working many hours.",
            "question": "Was her work easy?",
            "answer": "No, it was hard work."
        },
    ]),
]

sentiment = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['statement']}",
    output_formatter=lambda x: f"Sentiment: {x['sentiment']}",
    required_keys=["statement", "question"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Is the sentiment of the passage positive, negative, or neutral?\n\n"
)

sentiment_examples = [
    pd.DataFrame([
        {
            "statement": "Mary saw the animal",
            "sentiment": "neutral",
        },
        {
            "statement": "the town is quaint , but ultimately too boring and ugly",
            "sentiment": "negative",
        },
        {
            "statement": "he's a strong athlete, people come from miles away to watch him compete",
            "sentiment": "positive",
        },
    ]),
    pd.DataFrame([
        {
            "statement": "the movie was not very good, they could have picked a better lead actor",
            "sentiment": "negative",
        },
        {
            "statement": "she loves her mother so much that she gives her a hug everyday",
            "sentiment": "positive",
        },
        {
            "statement": "the computer sat on the table",
            "sentiment": "neutral",
        },
    ]),
    pd.DataFrame([
        {
            "statement": "Mary saw the animal",
            "sentiment": "neutral",
        },
        {
            "statement": "he's a strong athlete, people come from miles away to watch him compete",
            "sentiment": "positive",
        },
        {
            "statement": "the dress is boring and ugly, it loooks like a towel",
            "sentiment": "negative",
        },
        {
            "statement": "the exam went well since i studied a lot",
            "sentiment": "positive",
        },
        {
            "statement": "the table was made of wood",
            "sentiment": "neutral",
        },
        {
            "statement": "grocery stores sell food",
            "sentiment": "neutral",
        },
    ])
]

sentiment_more_positive = InputOutputPrompt(
    input_formatter=lambda x: f"Sentence 1: {x['sentence1']}\nSentence 2: {x['sentence2']}",
    output_formatter=lambda x: f"More positive: {x['answer']}",
    required_keys=["sentence1", "sentence2", 'answer'],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Which sentence is more positive?\n\n"
)

sentiment_more_positive_examples = [
    pd.DataFrame([
        {
            "sentence1": "i think she's fine",
            "sentence2": "she's my favorite person in the world",
            "answer": "she's my favorite person in the world"
        },
        {
            "sentence1": "i have never been to a better restaurant in my life",
            "sentence2": "the restaurant was decent, I may go back",
            "answer": "i have never been to a better restaurant in my life"
        },
        {
            "sentence1": "I went on the best vacation with my family last week.",
            "sentence2": "I just got back from a vacation, which was expensive, but fun",
            "answer": "I went on the best vacation with my family last week."
        }
    ])
]


what_next = InputOutputPrompt(
    input_formatter=lambda x: f"Choices:\n- {x['choice_a']}\n- {x['choice_b']}\n\nPassage: {x['passage']} Then?",
    output_formatter=lambda x: f"{x['answer']}",
    required_keys=["choice_a", "choice_b", "passage", "answer"],
    input_output_sep=" ",
    example_sep="\n\n----\n\n",
    instruction="Pick the best choice for the passage.\n\n"
)

what_next_examples = [
    pd.DataFrame([
        {
            "passage": "The girl went to college and graduated with honors",
            "choice_a": "She was qualified to get a job",
            "choice_b": "She was qualified to eat pizza",
            "answer": "she was qualified to get a job"
        },
        {
            "passage": "Max bought all his friends cupcakes for the party.",
            "choice_a": "They never spoke to him again",
            "choice_b": "They all thanked him",
            "answer": "They all thanked him"
        },
        {
            "passage": "Sam felt so hungry so he bought himself some cheese!",
            "choice_a": "he was starving",
            "choice_b": "he felt full",
            "answer": "he felt full"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "The girl went to college and graduated with honors",
            "choice_a": "She was qualified to get a job",
            "choice_b": "She was qualified to eat pizza",
            "answer": "she was qualified to get a job"
        },
        {
            "passage": "Max bought all his friends cupcakes for the party.",
            "choice_a": "They never spoke to him again",
            "choice_b": "They all thanked him",
            "answer": "They all thanked him"
        },
        {
            "passage": "Sam felt so hungry so he bought himself some cheese!",
            "choice_a": "he was starving",
            "choice_b": "he felt full",
            "answer": "he felt full"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Sam and Jade were excited to see the new movie.",
            "choice_a": "They went to the theater",
            "choice_b": "They went swimming",
            "answer": "They went to the theater"
        },
        {
            "passage": "Andy's parents got him a new toy",
            "choice_a": "he played",
            "choice_b": "he cried",
            "answer": "he played"
        },
        {
            "passage": "She can read the entire book in a single day.",
            "choice_a": "She is a slow reader",
            "choice_b": "She is a fast reader",
            "answer": "She is a fast reader"
        }
    ])
]

class StoryCloze(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def read_data(self, save_dir, overwrite_data):
        save_data = Path(f"{save_dir}/{self.task_name}/data.feather")
        if not save_data.exists() or overwrite_data:
            dataset = load_dataset("story_cloze", "2016", data_dir=self.data_dir)
            test_data = dataset[self.val_split].to_pandas()
            test_data.to_feather(f"{save_data}")
        else:
            print(f"Reading test data from {save_data}")
            test_data = pd.read_feather(f"{save_data}")

        save_data_train = Path(f"{save_dir}/{self.task_name}/train_data.feather")
        if not save_data_train.exists() or overwrite_data:
            dataset = load_dataset("story_cloze", "2016", data_dir=self.data_dir)
            train_data = dataset["validation"].to_pandas()
            train_data.to_feather(f"{save_data_train}")
        else:
            print(f"Reading train data from {save_data_train}")
            train_data = pd.read_feather(f"{save_data_train}")

        print(f"Test Data Size: {len(test_data)}")
        print(f"Train Data Size: {len(train_data)}")
        return test_data, train_data

    def get_boost_decomp_examples(self, data_train, boost_id):
        if boost_id < 3:
            return [
                questioner_examples[boost_id],
                openended_qa_examples[boost_id],
                sentiment_examples[boost_id],
                what_next_examples[boost_id],
                sentiment_more_positive_examples[0]
            ]
        else:
            seed = [1, 2, 3][boost_id-3]
            k_shot = 8  #32#4
            random.seed(seed)
            np.random.seed(seed)
            sub_df = data_train.sample(k_shot)
            booster_df = sub_df.sample(frac=1, random_state=0)
            print(f"Selected: {len(booster_df)} in context examples.")
            return [
                booster_df
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
                pred = entry["pred"]
                gold = entry["gold"]
            else:
                instruction = "Given two possible next sentences A) and B), choose the best next sentence to complete the story. Answer with A or B."
                icl_str = ""
                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        s_text = f"{s_row['input_sentence_1']} {s_row['input_sentence_2']} {s_row['input_sentence_3']} {s_row['input_sentence_4']}\n\n"
                        if s_row['answer_right_ending'] == 1:
                            answer = s_row['sentence_quiz1']
                        elif s_row['answer_right_ending'] == 2:
                            answer = s_row['sentence_quiz2']
                        choices = f"A) {s_row['sentence_quiz1']}\nB) {s_row['sentence_quiz2']}\n\n"
                        icl_str += f"{s_text}{choices}Answer: {answer}\n\n\n"
                
                text = f"{row['input_sentence_1']} {row['input_sentence_2']} {row['input_sentence_3']} {row['input_sentence_4']}\n\n"
                choices = f"A) {row['sentence_quiz1']}\nB) {row['sentence_quiz2']}\n\n"
                gold = ''
                if row['answer_right_ending'] == 1:
                    gold = row['sentence_quiz1']
                elif row['answer_right_ending'] == 2:
                    gold = row['sentence_quiz2']
                prompt = f"{instruction}\n\n\n{icl_str}{text}{choices}Answer: "

                raw_answer = get_response(prompt, manifest, max_toks=50) 
                answer = raw_answer.split("\n")
                answer = [a for a in answer if a]
                if answer:
                    answer = answer[0].replace("Answer: ", "").strip()
                else:
                    answer = ''

                if i == 0:
                    print(prompt)
                
                answer = answer.replace(")", "").replace("(", "").replace(":", "")
                is_A = answer.lower() in row['sentence_quiz1'].lower() or row['sentence_quiz1'].lower() in answer.lower() or "A" in answer.split()
                is_B = answer.lower() in row['sentence_quiz2'].lower() or row['sentence_quiz2'].lower() in answer.lower() or "B" in answer.split()
                pred = ''
                if is_A and (not is_B):
                    pred = '1'
                if is_B and (not is_A):
                    pred = '2'

                entry = {
                    "ind": ind,
                    "example": text,
                    "base_prompt": prompt,
                    "raw_answer": raw_answer,
                    "pred": pred,
                    "gold": gold,
                }
                expt_log[ind] = entry

            labels.append(str(row['answer_right_ending']))
            preds.append(pred)

        report = classification_report(labels, preds, output_dict=True)
        return expt_log, report["accuracy"]

    def get_question(self, statement, all_prompts, boost_examples, manifest, overwrite_manifest):
        questioner = all_prompts[0](boost_examples[0])

        question_prompt = f"{questioner}\n\nStatement: {{statement:}}\n"
        question = get_response(
            question_prompt.format(statement=statement), 
            manifest, 
            max_toks= 4*len(statement.split()))
        question = question.replace("Question: ", "")
        question = [q for q in question.split("\n") if q]
        if not question:
            question = f"{statement} Yes or no?"
        else:
            question = question[0]
        return question, question_prompt

    def answer_question(self, question, passage, all_prompts, boost_examples, manifest, overwrite_manifest, option=1):
        one_at_a_time = all_prompts[1](boost_examples[1])

        answer_prompt = f"{one_at_a_time}\n\nPassage: {{passage:}}\nQuestion: {{question:}}\n"
        answer = get_response(
            answer_prompt.format(passage=passage, question=question), 
            manifest, 
            max_toks=50)
        answer = answer.replace("Answer: ", "")
        answer = [a for a in answer.split("\n") if a]
        if answer:
            answer = answer[0].replace(",", "").replace(".", "").lower()
        else:
            answer = ''
        pred = ''
        if option == 1:
            if 'yes' in answer.split():
                pred = "1"
            elif 'no' in answer.split():
                pred = "2"
        elif option == 2:
            if 'no' in answer.split():
                pred = "1"
            elif 'yes' in answer.split():
                pred = "2"
        return pred, answer_prompt

    def get_one_by_one(self, example, choice_a, choice_b, all_prompts, boost_examples, manifest, overwrite_manifest):

        # construct questions
        question_a, questioner_prompt = self.get_question(choice_a, all_prompts, boost_examples, manifest, overwrite_manifest)
        question_b, questioner_prompt = self.get_question(choice_b, all_prompts, boost_examples, manifest, overwrite_manifest)

        # ask questions
        pred_a, answerer_prompt = self.answer_question(question_a, example, all_prompts, boost_examples, manifest, overwrite_manifest, option=1)
        pred_b, answerer_prompt = self.answer_question(question_b, example, all_prompts, boost_examples, manifest, overwrite_manifest, option=2)
        
        # reconcile answer
        if pred_a == "1" and pred_b == "1":
            pred = "1"
        elif pred_a == "2" and pred_b == "2":
            pred = "2"
        elif pred_a and not pred_b:
            pred = pred_a
        elif not pred_b and pred_b:
            pred = pred_b
        else:
            pred = ''
        return pred, questioner_prompt, answerer_prompt

    def get_sentiment(self, statement, all_prompts, boost_examples, manifest, overwrite_manifest):
        sentiment_prompt = all_prompts[0](boost_examples[2])
        prompt = f"{sentiment_prompt}\n\nPassage: {{statement:}}\nSentiment: "
        raw_answer = get_response(
            prompt.format(statement=statement), 
            manifest,
            max_toks=5)
        sent = raw_answer.split("\n")[0]

        if "positive" in sent:
            sent = 1
        elif "negative" in sent:
            sent = -1
        elif "neutral" in sent:
            sent = 0

        return sent, sentiment_prompt

    def get_sentiment_more_pos(self, choice_a, choice_b, all_prompts, boost_examples, manifest, overwrite_manifest):
        sentiment_prompt = all_prompts[1](boost_examples[4])
        prompt = f"{sentiment_prompt}\n\nSentence 1: {{choice_a:}}\nSentence 2: {{choice_b:}}\nMore positive:"
        raw_answer = get_response(
            prompt.format(choice_a=choice_a, choice_b=choice_b), 
            manifest,
            max_toks=20)
        raw_answer = raw_answer.split("\n")[0].lower()
        if choice_a.lower() in raw_answer and not choice_b.lower() in raw_answer:
            return 1
        elif choice_b.lower() in raw_answer and not choice_a.lower() in raw_answer:
            return 2
        else:
            return 0

    def combine_sentiments(self, example, choice_a, choice_b, all_prompts, boost_examples, manifest, boost_id, overwrite_manifest):

        # construct questions
        sentiment_a, sentiment_prompt = self.get_sentiment(choice_a, all_prompts, boost_examples, manifest, overwrite_manifest)
        sentiment_b, sentiment_prompt = self.get_sentiment(choice_b, all_prompts, boost_examples, manifest, overwrite_manifest)
        sentiment_ex, sentiment_prompt = self.get_sentiment(example, all_prompts, boost_examples, manifest, overwrite_manifest) 

        # reconcile answer
        pred = ''
        if abs(sentiment_a - sentiment_ex) < abs(sentiment_b - sentiment_ex):
            pred = "1"
        elif abs(sentiment_a - sentiment_ex) > abs(sentiment_b - sentiment_ex):
            pred = "2"

        return pred, sentiment_prompt

    def get_what_next(self, example, choice_a, choice_b, all_prompts, boost_examples, manifest, overwrite_manifest):
        what_next_prompt = all_prompts[0](boost_examples[3])
        prompt = f"{what_next_prompt}\n\n----\n\nChoices:\n- {{choice_a:}}\n- {{choice_b:}}\n\nPassage: {{example:}} Then?"
        raw_answer = get_response(
            prompt.format(choice_a=choice_a, choice_b=choice_b, example=example), 
            manifest,
            max_toks=50)
        answer = raw_answer.split("\n")[0].lower()
        choice_a = choice_a.lower()
        choice_b = choice_b.lower()
        pred = ''
        for n in range(5,50):
            for idx_offset in range(len(answer) - n + 1):
                ngram = "".join(answer[idx_offset:idx_offset+n])
                if ngram in choice_a and ngram not in choice_b:
                    pred = '1'
                elif ngram not in choice_a and ngram in choice_b:
                    pred = '2'
        return pred, what_next_prompt

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
        no_preds = 0

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            example = f"{row['input_sentence_1']} {row['input_sentence_2']} {row['input_sentence_3']} {row['input_sentence_4']}"
            choice_a = row['sentence_quiz1']
            choice_b = row['sentence_quiz2']
            gold = str(row['answer_right_ending'])

            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            for boost_num, boost_examples in enumerate(boost_dfs):
                
                all_prompts = []

                if boost_num < 3:
                    pred, questioner_prompt, answerer_prompt = self.get_one_by_one(
                        example, choice_a, choice_b, [questioner, openended_qa], boost_examples, manifest, overwrite_manifest
                    )
                    if i == 0:
                        print(questioner_prompt)
                        print("\n\n")
                        print(answerer_prompt)
                        print("\n\n")
                    all_prompts.append(questioner_prompt)
                    all_prompts.append(answerer_prompt)

                    if not pred:
                        pred, sentiment_prompt = self.combine_sentiments(
                            example, choice_a, choice_b, [sentiment, sentiment_more_positive], boost_examples, manifest, boost_num, overwrite_manifest
                        )
                        all_prompts.append(sentiment_prompt)

                    if not pred:
                        pred, what_next_prompt = self.get_what_next(
                            example, choice_a, choice_b, [what_next], boost_examples, manifest, overwrite_manifest
                        )
                        pred2, what_next_prompt = self.get_what_next(
                            example, choice_b, choice_a, [what_next], boost_examples, manifest, overwrite_manifest
                        )
                        if pred != pred2:
                            pred = ""
                        all_prompts.append(what_next_prompt)

                    if not pred:
                        pred = ''
                        no_preds += 1
                else:
                    icl_str = ""
                    for s_ind, s_row in boost_examples[0].iterrows():
                        s_text = f"{s_row['input_sentence_1']} {s_row['input_sentence_2']} {s_row['input_sentence_3']} {s_row['input_sentence_4']}"
                        if s_row['answer_right_ending'] == 1:
                            answer = s_row['sentence_quiz1']
                        elif s_row['answer_right_ending'] == 2:
                            answer = s_row['sentence_quiz2']
                        icl_str += f"Context: {s_text} {answer}\n\n"
                    
                    text = f"{row['input_sentence_1']} {row['input_sentence_2']} {row['input_sentence_3']} {row['input_sentence_4']}"
                    options = [row['sentence_quiz1'], row['sentence_quiz2']]
                    if row['answer_right_ending'] == 1:
                        gold = row['sentence_quiz1']
                    elif row['answer_right_ending'] == 2:
                        gold = row['sentence_quiz2']
                    prompt = f"Complete the paragraph.\n\n\n{icl_str}Context: {text}"
                    if i == 0:
                        print(prompt.format(text=text))
                    all_prompts.append(prompt)
                    raw_answer, _ = get_response(
                        prompt.format(text=text),
                        manifest,
                        gold_choices=[options[0].replace("- ", "").strip(), options[1].replace("- ", "").strip()],
                        overwrite=bool(overwrite_manifest),
                        max_toks=max(len(opt) for opt in options)*4,
                    )
                    
                    answer = raw_answer
                    is_A = answer.lower() in row['sentence_quiz1'].lower() or row['sentence_quiz1'].lower() in answer.lower() or "A" in answer.split()
                    is_B = answer.lower() in row['sentence_quiz2'].lower() or row['sentence_quiz2'].lower() in answer.lower() or "B" in answer.split()
                    pred = ''
                    if is_A and (not is_B):
                        pred = '1'
                    if is_B and (not is_A):
                        pred = '2'

                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)
            entry = {
                "ind": ind,
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "example": example,
                "choice_a": choice_a,
                "choice_b": choice_b,
                "gold": str(row['answer_right_ending']),
            }
            expt_log[ind] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(str(row['answer_right_ending']))
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 6
    task_name = "story_cloze"
    data_dir = f"{DATA_DIR}/story_cloze"
    decomp = StoryCloze(task_name, data_dir, val_split="test")
    decomp.run(args)


if __name__ == "__main__":
    main()
