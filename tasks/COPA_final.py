#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
from collections import Counter
import pandas as pd
import random
import numpy as np

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

what_next = InputOutputPrompt(
    input_formatter=lambda x: f"Question: {x['example']}",
    output_formatter=lambda x: f"{x['continue']}",
    required_keys=["example", "continue"],
    input_output_sep=" ",
    example_sep="\n\n",
    instruction="Pick the correct ending for the example.\n\n"
)

what_next_examples = [
    pd.DataFrame([
        {
            "example": "(because 'she took medicine', because 'she got expelled') My roommate was feeling better because?",
            "continue": "'she took medicine'",
        },
        {
            "example": "(because 'he does not practice', because 'he is fast') Matt is not good at soccer because?", 
            "continue": "'he does not practice'",
        },
        {
            "example": "(because 'she was smart', because 'she never did her homework') The girl went to college and graduated with honors because?", 
            "continue": "'she was smart'",
        },
    ]),
    pd.DataFrame([
        {
            "example": "(so 'he is always tired', so 'he is always sleeping') My dad works very hard so",
            "continue": "'he is always tired'",
        },
        {
            "example": "(so 'she threw a party', so 'she took medicine') My roommate was sick so", 
            "continue": "'she took medicine'",
        },
        {
            "example": "(so 'he played', so 'he cried') Andy's parents got him a new toy so", 
            "continue": "'he played'",
        },
    ]),
]

question = InputOutputPrompt(
    input_formatter=lambda x: f"Question: {x['example']}",
    output_formatter=lambda x: f"Answer: {x['continue']}",
    required_keys=["example", "continue"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Pick the correct ending for the example.\n\n"
)

question_examples = [
    pd.DataFrame([
        {
            "example": "What best continues the sentence \"My dad often talks about long hours at work because\"?",
            "continue": "\"work is hard\"",
        },
        {
            "example": "What best continues the sentence \"My roommate was sick and took medicine and so\"?", 
            "continue": "\"she felt better\"",
        },
        {
            "example": "What best continues the sentence \"Andy's parents got him a new toy and so\"?",
            "continue": "\"he played\"",
        },
        {
            "example": "What best continues the sentence \"My roommate was feeling better because\"?",
            "continue": "\"she took medicine\"",
        }
    ])
]

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
            "context": "It was a beautiful summer day outside. Bob decided to go for a walk at the park. Bob walked along the path and admired the scenery. He found a twenty dollar bill on the ground.",
            "question": "Was he disappointed?",
            "answer": "No, he was happy he got money."
        },
        {
            "context": "Mike is a busy man. He often eats fast food for breakfast. Mike wanted to enjoy a healthier breakfast. He tried an overnight oatmeal recipe.",
            "question": "Did Mike eat the oatmeal?",
            "answer": "Yes"
        },
        {
            "context": "Gina's sister cut her ankle on broken glass. The blood ran down her foot and into her shoe. When she saw the blood she ran home. Gina ran behind her, but couldn't keep up.",
            "question": "Did Gina's sister go to the doctor?",
            "answer": "Yes, because she was bleeding"
        },
    ]),
    pd.DataFrame([
        {
            "context": "My aunt is a nurse she works a lot. Last week was especially bad and she was constantly working many hours.",
            "question": "Was her work easy?",
            "answer": "No"
        },
        {
            "context": "It was a beautiful day outside. Bob decided to go for a walkk. Bob walked along the path and admired the scenery. He found a twenty dollar bill on the ground.",
            "question": "Was he disappointed?",
            "answer": "No, he was happy he got money."
        },
        {
            "context": "Mom didn't want to cook dinner tonight. We were all very hungry. She told us to fend for ourselves. We ate cold cereal for dinner tonight.",
            "question": "Was everyone upset about the dinner?",
            "answer": "Yes, the food was cold"
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
    ])
]

what_next2 = InputOutputPrompt(
    input_formatter=lambda x: f"Choices:\n- {x['choice_a']}\n- {x['choice_b']}\n\nPassage: {x['passage']}",
    output_formatter=lambda x: f"{x['answer']}",
    required_keys=["choice_a", "choice_b", "passage", "answer"],
    input_output_sep=" ",
    example_sep="\n\n----\n\n",
    instruction="Pick the best choice for the passage.\n\n"
)

what_next_examples2 = [
    pd.DataFrame([
        {
            "passage": "My dad often talks about long hours at work. Because?",
            "choice_a": "work is hard",
            "choice_b": "work is easy",
            "answer": "work is hard"
        },
        {
            "passage": "My roommate was sick and took medicine. So?",
            "choice_a": "she threw a party",
            "choice_b": "she felt better",
            "answer": "she felt better"
        },
        {
            "passage": "Andy's parents got him a new toy. So?",
            "choice_a": "he played",
            "choice_b": "he cried",
            "answer": "he played"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "The girl went to college and graduated with honors.",
            "choice_a": "She was qualified to get a job.",
            "choice_b": "She was qualified to eat pizza.",
            "answer": "she was qualified to get a job."
        },
        {
            "passage": "Max bought all his friends cupcakes for the party.",
            "choice_a": "They never spoke to him again.",
            "choice_b": "They all thanked him.",
            "answer": "They all thanked him."
        },
        {
            "passage": "Sam felt so hungry so he bought himself some cheese!",
            "choice_a": "After he ate the cheese, he was starving.",
            "choice_b": "After he ate the cheese, he felt better.",
            "answer": "After he ate the cheese, he felt better."
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Sam and Jade were excited to see the new movie.",
            "choice_a": "They went to the theater.",
            "choice_b": "They went swimming.",
            "answer": "They went to the theater."
        },
        {
            "passage": "Matt is very competitive in soccer.",
            "choice_a": "He practices all the time.",
            "choice_b": "He loves to lose.",
            "answer": "He practices all the time."
        },
        {
            "passage": "She can read the entire book in a single day.",
            "choice_a": "She is a slow reader.",
            "choice_b": "She is a fast reader.",
            "answer": "She is a fast reader."
        }
    ])
]


class COPADecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def zero_few_baseline(
        self,
        test_data,
        few_shot_df,
        manifest,
        overwrite_manifest,
        do_few_shot=True,
    ):
        expt_log = {}
        total = 0
        total_crct = 0 

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            if ind in expt_log:
                pred = entry["pred"]
                gold = entry["gold"]
            else:
                icl_str = ""

                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        s_text = s_row['inputs_pretokenized'].replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
                        s_parts = s_text.split(". ")
                        s_sentence = s_parts[0]
                        s_transition = s_parts[1]
                        options = [l for l in s_text.split("\n") if l.startswith("- ")]
                        if "as a consequence" in s_transition:
                            s_text = f"{s_sentence} so"
                        elif "as a result of" in s_transition:
                            s_text = f"{s_sentence} because"
                        icl_str += f"Context: {s_text} {s_row['targets_pretokenized']}\n\n"
                
                text = row['inputs_pretokenized']
                parts = text.split(". ")
                sentence = parts[0]
                transition = parts[1]
                options = [l for l in text.split("\n") if l.startswith("- ")]
                if "as a consequence" in transition:
                    text = f"{sentence} so"
                elif "as a result of" in transition:
                    text = f"{sentence} because"
                text = text.replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
                gold = row['targets_pretokenized']
                prompt = f"Pick the more likely continuation to the following sentence.\n\n\n{icl_str}Context: {{text:}}"
                pmp = prompt.format(text=text)
                if i == 0:
                    print(pmp)
                raw_answer, _ = get_response(
                    pmp,
                    manifest,
                    gold_choices=[options[0].replace("- ", "").strip(), options[1].replace("- ", "").strip()],
                    overwrite=bool(overwrite_manifest),
                    max_toks=50,
                )
                answer = raw_answer.strip().lower()
                answer = answer.split("\n")
                answer = [a for a in answer if a]
                if answer:
                    answer = answer[0].replace("-", "").strip()
                else:
                    answer = ''

                pred = "".join([a for a in answer if a not in [".", ",", "?", ";", ":", "'", '"']])
                gold = "".join([a for a in gold if a not in [".", ",", "?", ";", ":", "'", '"']])
                
                crct = gold.lower() == pred.lower()
                total += 1
                total_crct += crct

                entry = {
                    "ind": ind,
                    "example": text,
                    "base_prompt": pmp,
                    "raw_answer": raw_answer,
                    "pred": pred,
                    "gold": gold,
                }
                expt_log[ind] = entry

        accuracy = total_crct/total
        return expt_log, accuracy

    def get_boost_decomp_examples(self, train_data, boost_id):
        if boost_id < 1: 
            return [
                what_next_examples[boost_id],
            ]
        elif boost_id < 2:
            return [
                what_next_examples2[boost_id-1],
            ]
        elif boost_id >= 2:
            seed = [1, 2, 3][boost_id-2]
            k_shot = 4*seed
            random.seed(seed)
            np.random.seed(seed)

            data_train = pd.DataFrame(train_data)
            sub_df = data_train.sample(k_shot)
            booster_df = sub_df.sample(frac=1, random_state=0)
            print(f"Selected: {len(booster_df)} in context examples.")
            return [
                booster_df
            ]

    def what_happened_next(self, prompt, boost_ex, example, transition, choice_a, choice_b, word, manifest, overwrite_manifest):
        example = example.strip(".")
        choice_a = choice_a.lower()
        choice_b = choice_b.lower()
        transition = transition.strip()
        prompt_suffix = prompt(boost_ex)
        ex_prompt = f"{prompt_suffix}\n\nQuestion: ({{word:}} \'{{choice_a:}}\', {{word:}} \'{{choice_b:}}\') {{example:}} {{word:}}?"
        raw_answer = get_response(
            ex_prompt.format(word=word, choice_a=choice_a, choice_b=choice_b, example=example), 
            manifest,
            max_toks= 4*len(choice_a.split()),
            overwrite=bool(overwrite_manifest))
        answer = [q for q in raw_answer.split("\n") if q][0].lower()
        pred = ''
        for n in range(5,50):
            for idx_offset in range(len(answer) - n + 1):
                ngram = "".join(answer[idx_offset:idx_offset+n])
                if ngram in choice_a and ngram not in choice_b:
                    pred = choice_a
                elif ngram not in choice_a and ngram in choice_b:
                    pred = choice_b
        return pred, ex_prompt

    def question_answer(self, prompt, boost_ex, example, transition, choice_a, choice_b, word, manifest, overwrite_manifest):
        example = example.strip(".")
        choice_a = choice_a.lower()
        choice_b = choice_b.lower()
        transition = transition.strip()
        prompt_suffix = prompt(boost_ex)
        ex_prompt = f"{prompt_suffix}\n\nQuestion: What best continues the sentence \"{{example:}}\"?\nAnswer:"
        ex_pmp = ex_prompt.format(example=example)
        raw_answer, log_prob = get_response(
            ex_pmp, 
            manifest,
            gold_choices=[choice_a, choice_b],
            max_toks= 4*len(choice_a.split()),
            overwrite=bool(overwrite_manifest))
        answer = [q for q in raw_answer.split("\n") if q][0].lower()
        pred = ''
        for n in range(5,50):
            for idx_offset in range(len(answer) - n + 1):
                ngram = "".join(answer[idx_offset:idx_offset+n])
                if ngram in choice_a and ngram not in choice_b:
                    pred = '1'
                elif ngram not in choice_a and ngram in choice_b:
                    pred = '2'
        if not pred:
            import pdb;
            pdb.set_trace()
        return pred, ex_pmp

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
            pred = choice_a
        elif pred_a == "2" and pred_b == "2":
            pred = choice_b
        elif pred_a and not pred_b:
            if pred_a == "1":
                pred = choice_a
            else:
                pred = choice_b
        elif not pred_b and pred_b:
            if pred_b == "1":
                pred = choice_a
            else:
                pred = choice_b
        else:
            pred = ''
        return pred, questioner_prompt, answerer_prompt

    def get_sentiment(self, statement, all_prompts, boost_examples, manifest, overwrite_manifest):
        sentiment_prompt = all_prompts[0](boost_examples[0])
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

    def combine_sentiments(self, example, choice_a, choice_b, all_prompts, boost_examples, manifest, overwrite_manifest):

        # construct questions
        sentiment_a, sentiment_prompt = self.get_sentiment(choice_a, all_prompts, boost_examples, manifest, overwrite_manifest)
        sentiment_b, sentiment_prompt = self.get_sentiment(choice_b, all_prompts, boost_examples, manifest, overwrite_manifest)
        sentiment_ex, sentiment_prompt = self.get_sentiment(example, all_prompts, boost_examples, manifest, overwrite_manifest)
        
        # reconcile answer
        pred = ''
        if abs(sentiment_a - sentiment_ex) < abs(sentiment_b - sentiment_ex):
            pred = choice_a
        elif abs(sentiment_a - sentiment_ex) > abs(sentiment_b - sentiment_ex):
            pred = choice_b
        return pred, sentiment_prompt

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


    def get_what_next(self, example, choice_a, choice_b, transition, all_prompts, boost_examples, manifest, overwrite_manifest):
        what_next_prompt = all_prompts[0](boost_examples[0])
        if "result of":
            prompt = f"{what_next_prompt}\n\n----\n\nChoices:\n- {{choice_a:}}\n- {{choice_b:}}\n\nPassage: {{example:}} Because?"
        elif "consequence":
            prompt = f"{what_next_prompt}\n\n----\n\nChoices:\n- {{choice_a:}}\n- {{choice_b:}}\n\nPassage: {{example:}} So?"
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
                    pred = choice_a
                elif ngram not in choice_a and ngram in choice_b:
                    pred = choice_b
        return pred, what_next_prompt


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
            text = row['inputs_pretokenized']
            text = text.replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
            gold = row['targets_pretokenized']
            parts = text.split("\n")
            statement = parts[0].split(".")[0:-1]
            transition = parts[0].split(".")[-1]
            example = " ".join(statement)
            choice_a = parts[1].replace("-", "").strip()
            choice_b = parts[2].replace("-", "").strip()
            gold_idx = -1
            if gold.lower() == choice_a.lower():
                gold_idx = '1'
            else:
                gold_idx = '2'

            all_prompts = []

            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            for boost_num, boost_examples in enumerate(boost_dfs):
                icl_str = ""
                pred = ''
                answer2 = None

                if boost_num < 1:
                    all_prompts = []
                    if 'as a consequence' in transition:
                        answer, what_next_prompt = self.what_happened_next(question, boost_examples[0], example, transition, choice_a, choice_b, 'and so', manifest, overwrite_manifest)
                    else:
                        answer, what_next_prompt = self.what_happened_next(
                            question, boost_examples[0], example, transition, choice_a, choice_b, 'because', manifest, overwrite_manifest)

                    if 'as a consequence' in transition:
                        answer2, what_next_prompt = self.what_happened_next(question, boost_examples[0], example, transition, choice_b, choice_a, 'and so', manifest, overwrite_manifest)
                    else:
                        answer2, what_next_prompt = self.what_happened_next(
                            question, boost_examples[0], example, transition, choice_b, choice_a, 'because', manifest, overwrite_manifest)

                    if answer != answer2:
                        answer = ''

                    all_prompts.append(what_next_prompt)


                elif boost_num < 2:
                    answer, what_next_prompt = self.get_what_next(
                        example, choice_a, choice_b, transition, [what_next2], boost_examples, manifest, overwrite_manifest
                    )
                    answer2, what_next_prompt = self.get_what_next(
                        example, choice_b, choice_a, transition, [what_next2], boost_examples, manifest, overwrite_manifest
                    )
                    if answer != answer2:
                        answer = ''

                    all_prompts.append(what_next_prompt)
                    
                else:
                    icl_str = ""
                    for s_ind, s_row in boost_examples[0].iterrows():
                        s_text = s_row['inputs_pretokenized'].replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
                        s_parts = s_text.split(". ")
                        s_sentence = s_parts[0]
                        s_transition = s_parts[1]
                        options = [l for l in s_text.split("\n") if l.startswith("- ")]
                        if "as a consequence" in s_transition:
                            s_text = f"{s_sentence} so"
                        elif "as a result of" in s_transition:
                            s_text = f"{s_sentence} because"
                        s_gold = s_row['targets_pretokenized'].lower()
                        icl_str += f"Context: {s_text} {s_gold}\n\n"
                    
                    text = row['inputs_pretokenized']
                    parts = text.split(". ")
                    sentence = parts[0]
                    transition = parts[1]
                    options = [l.lower() for l in text.split("\n") if l.startswith("- ")]
                    if "as a consequence" in transition:
                        text = f"{sentence} so"
                    elif "as a result of" in transition:
                        text = f"{sentence} because"
                    text = text.replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
                    gold = row['targets_pretokenized']
                    prompt = f"Pick the more likely continuation to the following sentence.\n\n\n{icl_str}Context: {{text:}}"
                    if i == 0:
                        print(prompt.format(text=text))
                    all_prompts.append(prompt)
                    raw_answer, _ = get_response(
                        prompt.format(text=text),
                        manifest,
                        gold_choices=[options[0].replace("- ", "").strip(), options[1].replace("- ", "").strip()],
                        overwrite=bool(overwrite_manifest),
                        max_toks=50,
                    )
                    answer = raw_answer.strip().lower()
                    answer = answer.split("\n")
                    answer = [a for a in answer if a]
                    if answer:
                        answer = answer[0].replace("-", "").strip()
                    else:
                        answer = ''

                pred = "".join([a for a in answer if a not in [".", ",", "?", ";", ":", "'", '"']]).lower()
                gold = "".join([a for a in gold if a not in [".", ",", "?", ";", ":", "'", '"']]).lower()

                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)

            
            preds_across_boost.reverse()
            mapped_p = []
            for p in preds_across_boost:
                if not p:
                    mapped_p.append("")
                    continue
                if p == gold:
                    mapped_p.append(gold_idx)
                elif gold_idx == "1":
                    mapped_p.append("2")
                else:
                    mapped_p.append("1")

            all_boost_preds.append(mapped_p)

            entry = {
                "ind": ind,
                "example": text,
                "prompts": prompts_across_boost,
                "preds_boost": mapped_p,
                "gold": gold_idx,
            }
            expt_log[ind] = entry
            
            labels.append(gold_idx)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 5
    task_name = "super_glue_copa"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_copa_more_likely/"
    decomp = COPADecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
