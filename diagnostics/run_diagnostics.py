""" Running and Scoring the AMA Diagnostics """

import os
import json
from collections import Counter
from tqdm import tqdm
import openai
from manifest import Manifest
openai.api_key = "" # Find this on the OpenAI Website

from datasets import load_metric
rouge = load_metric("rouge")

######################### HELPER FUNCTIONS  #########################

def rougeL(preds, labels):
    return rouge.compute(predictions=preds, references=labels)['rougeL'].mid.fmeasure

from collections import Counter
def text_f1(preds=[], labels=[]):
    """Compute average F1 of text spans.

    Taken from Squad without prob threshold for no answer.
    """
    total_f1 = 0
    for pred, gold in zip(preds, labels):
        pred_toks = pred.split()
        gold_toks = gold.split()
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            total_f1 += int(gold_toks == pred_toks)
        elif num_same == 0:
            total_f1 += 0
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            total_f1 += f1
    f1_avg = total_f1 / len(labels)
    return f1_avg

def get_response(manifest, prompt, max_toks=10, temperature = 0, gold_choices=[], model_name="manifest", engine="text-davinci-002", logit_bias={}):
    prompt = prompt.strip()
    if model_name == 'openai':
        if logit_bias:
            completion = openai.Completion.create(engine=engine, prompt=prompt, temperature=temperature, top_p=1, max_tokens=max_toks, logprobs=5, logit_bias=logit_bias)
        else:
            completion = openai.Completion.create(engine=engine, prompt=prompt, temperature=temperature, top_p=1, max_tokens=max_toks, logprobs=5)
        response = completion.choices[0].text
    
    if model_name == "manifest":
        if gold_choices:
            max_len = max([len(g.split()) for g in gold_choices])
        else:
            max_len = 0
        max_token_args = ({"max_tokens": min(max_toks,8 * len(max_len),)}
            if gold_choices is None
            else {}
        )
        if gold_choices:
            response = manifest.run(prompt, gold_choices=gold_choices,overwrite_cache=False,**max_token_args,)
        else:
            response = manifest.run(prompt, max_tokens=max_toks, overwrite_cache=False)
        
    return response

def get_manifest_session(
    client_name="huggingface",
    client_connection="http://127.0.0.1:5000",
    cache_connection=None,
    temperature=0,
    top_p=1.0,
):
    if client_name == "huggingface" and temperature == 0:
        params = {
            "temperature": 0.001,
            "do_sample": False,
            "top_p": top_p,
        }
    elif client_name in {"openai", "ai21"}:
        params = {
            "temperature": temperature,
            "top_p": top_p,
        }
    else:
        raise ValueError(f"{client_name} is not a valid client name")
    manifest = Manifest(
        client_name=client_name,
        client_connection=client_connection,
        cache_name="sqlite",
        cache_connection=cache_connection,
        **params,
    )
    model_name = manifest.client.get_model_params()["model_name"]
    return manifest, model_name

######################### SCORE FUNCTIONS  #########################


"""  A "blah" maps to category: blah """
def selection_easy(dataset, manifest):
    preds = []
    for i, (ind, row) in tqdm(enumerate(dataset.items())):
        prefix = row['input']
        pred = get_response(manifest, prefix, max_toks=50)
        pred = [p for p in pred.split("\n") if p][0].strip()
        preds.append(pred)
        
        if i == 0:
            print(prefix)
            print(f"PRED: {pred}")
        
    labels = [l.strip() for l in row['labels']]
    num_valid = [p for p in preds if p in labels]
    return len(num_valid)/len(dataset)


"""  Does the model pick one of the given choices? """
def selection_hard(dataset, manifest):
    preds = []
    for i, (ind, row) in tqdm(enumerate(dataset.items())):
        prefix = row['input']
        pred = get_response(manifest, prefix, max_toks=50)
        pred = [p for p in pred.split("\n")][0]
        preds.append(pred)
        
        if i == 0:
            print(prefix)
            print(f"PRED: {pred}")
    
    valid = 0
    for (ind, row), pred in zip(dataset.items(), preds):
        choices = row['output']
        if pred.lower().strip(".") in [c.lower().strip(".") for c in choices]:
            valid += 1
    return valid/len(dataset)


"""  Does the model generate three choices? """
def text_generation(dataset, manifest):
    preds = []
    for i, (ind, row) in tqdm(enumerate(dataset.items())):
        prefix = row['input']
        pred = get_response(manifest, prefix, max_toks=50)
        pred = pred.split("\n\n")[0]
        pred = pred.split("\n")
        pred = list(set([a.replace("- ", "").strip() for a in pred]))
        preds.append(pred)
        
        if i == 0:
            print(prefix)
            print(f"PRED: {pred}")
        
    valid = 0
    for pred in preds:
        if len(pred) == 2:
            valid += 1
    return valid/len(dataset)


"""  Does the model faithfully transform the statement to a question? """
def question_generation(dataset, manifest):
    preds = []
    for i, (ind, row) in tqdm(enumerate(dataset.items())):
        prefix = row['input']
        pred = get_response(manifest, prefix, max_toks=50)
        pred = [p for p in pred.split("\n")][0]
        preds.append(pred)
        
        if i == 0:
            print(prefix)
            print(f"PRED: {pred}")
    
    outputs = [row['output'] for ind, row in dataset.items()]
    score = rougeL(preds=preds, labels = outputs)
    return score


"""  Does the model faithfully choose the sentence with the entity name? """
"""  Does the model faithfully answer given a keyword for extraction? """
def extraction(dataset, manifest):
    preds = []
    for i, (ind, row) in tqdm(enumerate(dataset.items())):
        prefix = row['input']
        pred = get_response(manifest, prefix, max_toks=50)
        pred = [p for p in pred.split("\n")][0]
        preds.append(pred)
        
        if i == 0:
            print(prefix)
            print(f"PRED: {pred}")
    
    outputs = [row['output'] for ind, row in dataset.items()]
    score = text_f1(preds=preds, labels = outputs)
    return score
    

def main():
    data_dir = "data/"
    synthetics =  {
        'selection_easy':selection_easy, 
        'selection_hard':selection_hard, 
        'extraction':extraction,
        "text_generation": text_generation,
        "question_generation": question_generation
    }
    
    manifest, model_name = get_manifest_session()
    
    synthetic_scores = {}
    for synthetic, function in synthetics.items():
        print(f"RUNNING {synthetic}")
        with open(f"{data_dir}/{synthetic}.json") as f:
            dataset = json.load(f)
        score = function(dataset, manifest)
        synthetic_scores[synthetic] = score
        print(f"SCORE: {score}")
    
    print(synthetic_scores)
    model_name = model_name.replace("/", "_") 
    
    if not os.path.exists(f"results/"):
        os.makedirs(f"results/")
    
    with open(f"results/{model_name}_results.json", "w") as f:
        json.dump(synthetic_scores, f)
        
    print(f"Saved to: results/{model_name}_results.json")
    
if __name__ == "__main__":
    main()
