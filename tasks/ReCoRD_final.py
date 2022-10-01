#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm

from pathlib import Path
from collections import Counter
import re
import pandas as pd
import json
import unicodedata
import string

from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

cloze_completion = InputOutputPrompt(
    input_formatter=lambda x: f"Context: {x['context']}",
    output_formatter=lambda x: f"{x['answer']}",
    input_output_sep=" ",
    example_sep="\n\n----\n\n",
    required_keys=["context"],
    instruction="Complete the paragraph.\n\n"
)

cloze_completion_examples = [
    pd.DataFrame([
        {
            "context": "Barack Hussein Obama is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African-American president of the United States. Obama previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.",
            "answer": "Obama was senator of the state of Illinois prior to becoming a US president."
        },
        {
            "context": "(CNN) -- Saif al-Islam Gadhafi, 38, has never lived a day in which his father Moammar didn't rule Libya -- as its undisputed leader inside the country and an enigmatic, controversial voice for the world. And yet, as the Libyan government faced a stiff popular uprising, it was Moammar Gadhafi's second eldest son -- and not the Leader of the Revolution himself -- who was first to talk to the nation about the unrest and detail a plan to address it. The speech, made early Monday on Libyan state television, does not mean that Saif Gadhafi has usurped power from his father: Senior U.S. officials said there's no indication the elder Gadhafi is losing his grip.Saif al-Islam Gadhafi, 38, gives Libya's first public speech acknowledging unrest. There's been no public indication why he, and not his father Moammar, talked.",
            "answer": "Even while some may see the son as more open to change, there's little question that his loyalty remains first with Moammar and that his father has given little indication publicly that he's ready to let go and calls the shots."
        },
        {
            "context": "The Beatles were an English rock band, formed in Liverpool in 1960, that comprised John Lennon, Paul McCartney, George Harrison and Ringo Starr. They are regarded as the most influential band of all time and were integral to the development of 1960s counterculture and popular music's recognition as an art form. They were led by primary songwriters Lennon and McCartney.",
            "answer": "It is without a doubt that the Beatles were influential in rock and roll."
        }
    ]),
    pd.DataFrame([
        {
            "context": "One of the Internet's great promises is that it's the ultimate democratizer. It's open to everyone and allows all people to communicate. Facebook and Google have added new translation tools, but they take different approaches.",
            "answer": "Pros and cons: Google's computerized approach means it can translate tons of content -- and fast."
        },
        {
            "context": "Los Angeles, often referred to by its initials L.A., is the largest city in the U.S. state of California. With a population of roughly 3.9 million as of 2020, it is the second largest city in the United States after New York City and one of the world's most populous megacities. ",
            "answer": "Los Angeles is known for its Mediterranean climate, ethnic and cultural diversity, Hollywood film industry, and sprawling metropolitan area."
        },
        {
            "context": "The United States is cutting funding to the U.N. education and science agency UNESCO after the agency voted to accept a Palestinian bid for full membership, the U.S. State Department said Monday. \"Today's vote by the member states of UNESCO to admit Palestine as member is regrettable, premature and undermines our shared goal of a comprehensive just and lasting peace in the Middle East,\" said State Department spokeswoman Victoria Nuland.",
            "answer": "Israel believes that the correct and only way to make progress in the diplomatic process with the Palestinian is through direct negotiations without preconditions."
        },
    ]),
    pd.DataFrame([
        {
            "context": "(CNN) -- Martin Luther King Jr. fought and died so blacks would no longer be viewed as inferior but rather enjoy the same inherent rights given to whites in America. Yet in 2014, 50 years since the passage of the Civil Rights Act, the West View News thinks it's appropriate to publish a story about our first black president, Barack Obama, with the headline, \"The Nigger in the White House.\" Oh, the times we are living in. ",
            "answer": "The entire incident shows how far America has to yet to go in race relations."
        },
        {
            "context": "Martin Skrtel has warned Liverpool’s top-four rivals they should be ready to fight as they look to take momentum from his ‘most important’ goal. The Slovakian defender had eight staples put into a head wound following a clash with Olivier Giroud but carried on with a bandage to score the equaliser deep into injury time in Sunday’s dramatic 2-2 draw with Arsenal at Anfield. Liverpool have a chance to push on over Christmas with a fixture away to Burnley followed by home games against Swansea and Leicester.",
            "answer": "The Liverpool defender celebrates his last-minute goal as the crowd go wild at Anfield"
        },
        {
            "context": "Tripoli, Libya (CNN) -- It has been almost two weeks since Eman al-Obeidy burst into our hotel in Tripoli, desperate for the world to hear her story of rape and torture. We were finally able to speak to her Wednesday, against the explicit wishes of the Libyan government. The interview with al-Obeidy was facilitated by Gadhafi's son Saadi. We asked al-Obeidy if she would be willing to come to Saadi Gadhafi's office. She agreed and Gadhafi sent a car to pick her up. She says she wants to clear her name, smeared on state TV. Story of rape and torture became known after she burst into a Tripoli hotel",
            "answer": "Later Saadi Gadhafi told me: \"The people responsible for raping her should face charges.\""
        },
    ])
]

# TODO: how to fit in more ICL-demos


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(labels, predictions):
    f1 = exact_match = total = 0
    for label_set, pred in zip(labels, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, pred, label_set)
        f1 += metric_max_over_ground_truths(f1_score, pred, label_set)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

class ReCoRDDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)
        # Download from https://sheng-z.github.io/ReCoRD-explorer/
        if not Path(f"{DATA_DIR}/record").exists():
            raise ValueError(f"{DATA_DIR}/record doesn't exist. Please download.")
        data = json.load(open(f"{DATA_DIR}/record/record_dev.json"))["data"]
        self.test_query2labels = {}
        t = 0
        for ex in data:
            passage = ex["passage"]["text"]
            passage = unicodedata.normalize("NFKD", passage)
            for q in ex["qas"]:
                t += 1
                answers = list(set([e["text"] for e in q["answers"]]))
                query = unicodedata.normalize("NFKD", q["query"].strip())
                key = (passage, query)
                if key in self.test_query2labels:
                    assert set(self.test_query2labels[key]) == set(answers)
                self.test_query2labels[key] = answers
        data = json.load(open(f"{DATA_DIR}/record/record_train.json"))["data"]
        self.train_query2labels = {}
        t2 = 0
        for ex in data:
            passage = ex["passage"]["text"]
            passage = unicodedata.normalize("NFKD", passage)
            for q in ex["qas"]:
                t2 += 1
                answers = list(set([e["text"] for e in q["answers"]]))
                query = unicodedata.normalize("NFKD", q["query"].strip())
                key = (passage, query)
                if key in self.train_query2labels:
                    assert set(self.train_query2labels[key]) == set(answers)
                self.train_query2labels[key] = answers
        print(f"Loaded {t} test examples and {t2} train examples")

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
            icl_str = ""
            if do_few_shot:
                for s_ind, s_row in few_shot_df.iterrows():
                    s_text = " ".join(s_row['inputs_pretokenized'].split("\n\n")[1:]).strip()
                    s_query = s_text.split("\n")[-1]
                    s_text = s_text.replace(s_query, "").strip().replace("@highlight", "").replace("\n\n", ". ")
                    
                    s_answer_choices = s_row['answer_choices']
                    s_clean_choices = []
                    for choice in s_answer_choices:
                        other_choices = [c for c in s_answer_choices if c != choice]
                        if not any(c for c in other_choices if choice.lower() in c.lower()):
                            s_clean_choices.append(s_query.replace("@placeholder", choice))
                            
                    s_answer = s_row['targets_pretokenized']
                    s_choices = "\n- ".join(list(s_clean_choices))
                    s_answer = s_query.replace("@placeholder", f'{s_answer}')
                    if s_ind + 1 == len(few_shot_df):
                        icl_str += f"Context: {s_text}\n\nAnswer: {s_answer}"
                    else:
                        icl_str += f"Context: {s_text}\n\nAnswer: {s_answer}\n\n----\n\n"
            
            text = " ".join(row['inputs_pretokenized'].split("\n\n")[1:]).strip()
            query = text.split("\n")[-1]
            passage = text.rsplit("\n", 1)[0]
            key = (passage, query.strip())
            if key in self.test_query2labels:
                golds = self.test_query2labels[key]
            else:
                golds = [row['targets_pretokenized']]
            text = text.replace(query, "").strip().replace("@highlight", "").replace("\n\n", ". ")
            # gold = row['targets_pretokenized']
            answer_choices = row['answer_choices']
            answer, prompt = self.get_final_answer_full_sentence(answer_choices, None, None, text, query, manifest, icl_str=icl_str)

            pred = answer
            entry = {
                "ind": ind,
                "example": text,
                "base_prompt": prompt,
                "raw_answer": answer,
                "pred": pred,
                "gold": golds,
            }
            expt_log[ind] = entry
            preds.append(pred)
            labels.append(golds)

        metrics = evaluate(labels, preds)
        print(metrics)
        return expt_log, metrics["exact_match"]

    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            cloze_completion_examples[boost_id]
        ]

    def get_final_answer_full_sentence(self, answer_choices, prompt, boost_ex, text, query, manifest, icl_str='', size=2):

        if boost_ex is None:
            answer_prefix = "\n\nAnswer: "
            prompt_suffix = icl_str
        else:
            answer_prefix = " "
            prompt_suffix = prompt(boost_ex)

        left, right = query.split("@placeholder")
        clean_choices = []
        for choice in answer_choices:
            clean_choices.append(f"{choice}{right}")
        #     other_choices = [c for c in answer_choices if c != choice]
        #     if not any(c for c in other_choices if choice.lower() in c.lower()):
        #         clean_choices.append(query.replace("@placeholder", choice))
        prompt = f"{prompt_suffix}\n\n----\n\nContext: {{text:}}{answer_prefix}{{left:}}"
        pmp = prompt.format(text=text, left=left)
        answers = []
        for choice_group in range(0, len(clean_choices), size):
            try:
                raw_answer, score = get_response(
                    pmp, 
                    manifest, 
                    max_toks=20, 
                    gold_choices=list(clean_choices[choice_group:choice_group+size])
                )
                raw_answer = raw_answer.replace(right, "").strip()
            except Exception as e:
                print(e)
                raw_answer = ""
                score = -1000
            answers.append((raw_answer, score))
        answers = sorted(answers, key=lambda x: x[1], reverse=True)
        final_answer = answers[0][0].strip()
           
        return final_answer, pmp


    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest, run_limit=-1)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest, run_limit=1000, is_train=True)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train)
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            individual_accuracies.append(evaluate(labels, [p[i] for p in all_boost_preds])["exact_match"])
        report = evaluate(labels, preds)
        return expt_log, expt_log_train, report["exact_match"], individual_accuracies

    def _run_decomp_single_data(self, test_data, boost_dfs, manifest, overwrite_manifest, run_limit=-1, is_train=False):
        expt_log = {}
        all_boost_preds = []
        all_boost_answers = []
        labels = []
        label_data = self.test_query2labels if not is_train else self.train_query2labels

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            text = " ".join(row['inputs_pretokenized'].split("\n\n")[1:]).strip()
            query = text.split("\n")[-1]
            passage = text.rsplit("\n", 1)[0]
            key = (passage, query.strip())
            if key in label_data:
                golds = label_data[key]
            else:
                golds = [row['targets_pretokenized']]
            answer_choices = row['answer_choices']

            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            answers_across_boost= []
            for boost_num, boost_examples in enumerate(boost_dfs):
                all_prompts = []
                if "@highlight" not in boost_examples[0]:
                    text = text.replace(query, "").strip().replace("@highlight", "").replace("\n\n", ". ")
                
                final_answer, prompt = self.get_final_answer_full_sentence(answer_choices, cloze_completion, boost_examples[0], text, query, manifest)

                if i == 0:
                    print(prompt)
                all_prompts.append(prompt)

                pred = final_answer

                answers_across_boost.append(final_answer)
                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)

            all_boost_preds.append(preds_across_boost)
            all_boost_answers.append(answers_across_boost)
            entry = {
                "ind": ind,
                "example": text,
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "gold": golds,
            }
            expt_log[ind] = entry
            labels.append(golds)

        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    task_name = "super_glue_record"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_record_exercise/"
    decomp = ReCoRDDecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
