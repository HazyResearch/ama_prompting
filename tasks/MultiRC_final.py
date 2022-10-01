#!/usr/bin/env python
# coding: utf-8
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

answer_prompt = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['passage']}\nQuestion: {x['question']}\n{x['choice_question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["passage", "question", "choice_question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Answer if the possible answer is a correct answer to the question.\n\n"
)
answer_prompt_examples = [
    pd.DataFrame([
        {
            "passage": "Sara wanted to play on a baseball team. She had never tried to swing a bat and hit a baseball before. Her Dad gave her a bat and together they went to the park to practice. Sara wondered if she could hit a ball. She wasn't sure if she would be any good. She really wanted to play on a team and wear a real uniform. She couldn't wait to get to the park and test out her bat. When Sara and her Dad reached the park, Sara grabbed the bat and stood a few steps away from her Dad. Sara waited as her Dad pitched the ball to her. Her heart was beating fast. She missed the first few pitches. She felt like quitting but kept trying. Soon she was hitting the ball very far. She was very happy and she couldn't wait to sign up for a real team. Her Dad was very proud of her for not giving up. ",
            "question": "Based on the previous passage, Who pitched the ball to Sara and where did it occur? ",
            "choice_question": "Is \"Her dad did in the park\" a correct answer?",
            "answer": "yes",
        },
        {
            "passage": "The Vice President stated that he called the President to discuss the rules of engagement for the CAP. He recalled feeling that it did no good to establish the CAP unless the pilots had instructions on whether they were authorized to shoot if the plane would not divert. He said the President signed off on that concept. The President said he remembered such a conversation, and that it reminded him of when he had been an interceptor pilot. The President emphasized to us that he had authorized the shootdown of hijacked aircraft. The Vice President's military aide told us he believed the Vice President spoke to the President just after entering the conference room, but he did not hear what they said. Rice, who entered the room shortly after the Vice President and sat next to him, remembered hearing him inform the President, \"Sir, the CAPs are up. Sir, they're going to want to know what to do.\" Then she recalled hearing him say, \"Yes sir.\" She believed this conversation occurred a few minutes, perhaps five, after they entered the conference room. We believe this call would have taken place sometime before 10:10 to 10:15. Among the sources that reflect other important events of that morning, there is no documentary evidence for this call, but the relevant sources are incomplete. Others nearby who were taking notes, such as the Vice President's chief of staff, Scooter Libby, who sat next to him, and Mrs. Cheney, did not note a call between the President and Vice President immediately after the Vice President entered the conference room. At 10:02, the communicators in the shelter began receiving reports from the Secret Service of an inbound aircraft-presumably hijacked-heading toward Washington. That aircraft was United 93. The Secret Service was getting this information directly from the FAA. The FAA may have been tracking the progress of United 93 on a display that showed its projected path to Washington, not its actual radar return. Thus, the Secret Service was relying on projections and was not aware the plane...",
            "question": "Based on the previous passage, Why was the Secret Service's information about United 93 flawed? ",
            "choice_question": "Is \"The Secret Service Didn't have access to FAA information\" a correct answer?",
            "answer": "no",
        },
        {
            "passage": "Patricia Cross and her boyfriend Larry Osborne , two students in a San Francisco school , become expelled for the publication of an off-campus underground paper .  As a result , a philosophy professor , Dr. Jonathon Barnett , resigns his teaching position and decides to become an advocate for the counterculture youth movement and , specifically , the use of LSD .  The hippies of the Haight-Ashbury district first see him as a hero and then as something even more .  Dr. Barnett even makes an appearance on the Joe Pyne TV show to voice his support of the hippie community and the use of LSD .  One scheming young man sees the opportunity to build Dr. Barnett as the head of a cult centered around the use of LSD .  He hopes to earn profit from the users , Dr. Barnett's speeches known as `` happenings , '' and their lifestyles .  At a massive LSD-fueled dance , Patricia begins to have a bad trip Which leads to an argument between her and Pat , ultimately splitting the couple up .  After Patricia realizes that she's pregnant , Dr. Barnett advises her to have an abortion , ultimately leading to Patricia attempting suicide .  However , Larry saves her and makes the destruction of Dr. Barnett's cult his primary objective .  Larry shoots Dr. Barnett from the crowd at one of his massive speeches .  As another hippie in attendance calms the audience and Elliot sees his new leader for their cult-like organization , Larry realizes that his assassination of Dr. Barnett simply made him a martyr for the hippie movement . ",
            "question": "Based on the previous passage, Why did Dr. Barnett resign from teaching? ",
            "choice_question": "Is \"Patricia expulsion\" a correct answer?",
            "answer": "yes",
        },
        {
            "passage": "I wondered if that were my case--if I rode out for honour, and not for the pure pleasure of the riding.  And I marvelled more to see the two of us, both lovers of one lady and eager rivals, burying for the nonce our feuds, and with the same hope serving the same cause.  We slept the night at Aird's store, and early the next morning found Ringan.  A new Ringan indeed, as unlike the buccaneer I knew as he was unlike the Quaker.  He was now the gentleman of Breadalbane, dressed for the part with all the care of an exquisite.  He rode a noble roan, in his Spanish belt were stuck silver-hafted pistols, and a long sword swung at his side.  When I presented Grey to him, he became at once the cavalier, as precise in his speech and polite in his deportment as any Whitehall courtier.  They talked high and disposedly of genteel matters, and you would have thought that that red-haired pirate had lived his life among proud lords and high-heeled ladies.  That is ever the way of the Highlander.  He alters like a clear pool to every mood of the sky, so that the shallow observer might forget how deep the waters are. ",
            "question": "Based on the previous passage, Who is described as both buccaneer and cavalier? ",
            "choice_question": "Is \"Quaker\" a correct answer?",
            "answer": "no"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Earth processes have not changed over time. The way things happen now is the same way things happened in the past. Mountains grow and mountains slowly wear away. The same process is at work the same as it was billions of years ago. As the environment changes, living creatures adapt. They change over time. Some organisms may not be able to adapt. They become extinct. Becoming extinct means they die out completely. Some geologists study the history of the Earth. They want to learn about Earths past. They use clues from rocks and fossils. They use these clues to make sense of events. The goal is to place things in the order they happened. They also want to know how long it took for those events to happen. ",
            "question": "Based on the previous passage, Who studies in order to learn about the earth's past? ",
            "choice_question": "Is \"Librarians\" a correct answer?",
            "answer": "no",
        },
        {
            "passage": "Sam Farragut is a sociopathic business executive in Southern California who forces a team of advertising agency employees to embark on a dangerous dirtbike trip to the Baja California desert in order to compete for his business .  The men are Warren Summerfield , a suicidal middle-aged ad executive who has been fired from the agency ; the straightlaced Paul McIlvain who is inattentive to his wife , and brash art designer Maxon who feels suddenly trapped after his girlfriend announces she is pregnant .  There are numerous long sequences of motorcycle riding on desert backroads .  Summerfield has been having an affair with McIlvian's wife .  He has not told his wife that he was fired and is simply serving out his tenure at the agency while looking for a new position .  His wife is actually aware of the affair .  Farragut convinces the ad men to make the motorcycle journey on the pretext of looking for a location to shoot a commercial .  In reality , Farragut is reckless and looking to involve the men in spontaneous edgy adventure of his own manipulation .  After they leave , McIlvain's wife suspects that Summerfield is planning to kill himself for the insurance money , but she can not convince Summerfield's wife to instigate a search .  The four men travel deeper into Mexico on isolated dirt roads .  At one point Summerfield contemplates plunging off a cliff .  After being humiliated by a young American couple in a Baja bar , Farragut tracks them down on the beach while accompanied by Maxon . ",
            "question": "Based on the previous passage, Under what pretext does a sociopathic company executive organize the motorcycle trip? ",
            "choice_question": "Is \"Because he wants to compete for his business, so he suggest looking for a place to shoot a commercial\" a correct answer?",
            "answer": "yes",
        },
        {
            "passage": "The mighty fane, with its three massive towers, rises majestically over the red roofs of the town. Its most striking feature is the great Norman screen, running up without buttresses or projections to the parapet and hiding the bases of the square, richly decorated towers of the west front. The plain centre of the screen is the work of Remigius, the first bishop. The rest of it is relieved with rich arcading of Late Norman and Early English periods. The wooden spires which crowned the towers were removed in 1807. In 1192 Hugh of Avalon determined to rebuild the Norman building of Remigius, which an earthquake had shaken. To him we owe the choir and eastern transept. His successors completed the western transept and began the west end of the nave. So much money had to be spent in rebuilding the central tower, which fell in 1239, that the canons could not rebuild the nave entirely, but had to incorporate the Norman end by Remigius. Unfortunately the axis of the west front does not correspond to that of the nave, which is too wide for its height. The low vaulting is a serious defect in the choir built by St. Hugh, but of the superb beauty of the Angel Choir, which encloses his shrine, there can be no doubt. In its richness of sculpture it is one of the masterpieces of Gothic architecture in England. The interior of the cathedral is remarkable for the harmony of its style, which is Lancet-Gothic, and the dim lighting of the nave only adds to its impressiveness. ",
            "question": "Based on the previous passage, Who was responsible for initially building the choir and eastern transept and in what year did he start? ",
            "choice_question": "Is \"It wasn't the Hugh of Avalon\" a correct answer?",
            "answer": "no",
        },
        {
            "passage": "If you beat a dog in Schuylkill County, you'll probably get a $100 fine. If you repeatedly beat a woman, you'll probably get the same fine. In 2001, county judges heard 98 Protection From Abuse cases, finding the defendant guilty in 48 percent of those cases, either after a hearing or through a technical violation or plea. Of those found guilty, the majority were ordered to pay court costs, plus a $100 fine. No defendants were ordered to pay more than a $250 fine for violating the court order. In 27 percent of the cases, the charges were dismissed or the defendant was found not guilty. In the rest of the cases, charges were withdrawn or the matter is not yet resolved. Sarah T. Casey, executive director of Schuylkill Women in Crisis, finds it disturbing that in most cases, the fine for violating a PFA is little more than the fine someone would get for cruelty and abuse toward an animal. \"In most of the counties surrounding Schuylkill County, the penalties given for indirect criminal contempt are much stiffer than those in Schuylkill County,\" Casey said. \"What kind of message are we sending those who repeatedly violate Protection From Abuse orders? That it's OK to abuse women in Schuylkill County, because you'll only get a slap on the wrist?\" Under state law, the minimum fine for contempt of a PFA is $100; the maximum fine is $1,000 and up to six months in jail. Like others who are familiar with how the county's legal system does and doesn't work for victims of domestic violence, Casey believes some changes are in order. Valerie West, a manager/attorney with Mid-Penn Legal Services, with offices in Pottsville and Reading, regularly handles domestic violence cases. She finds fault with the local requirement that a custody order must be established within 30 days after a PFA is filed. West said she feels a custody order should be allowed to stand for the full term of the PFA - up to 18 months - as it does in many other counties in the state. \"It places an undue burden on the plaintiff, in terms of cost, finding...",
            "question": "Based on the previous passage, What solution is West offering and how is it different for a plaintiff from what is already being practiced? ",
            "choice_question": "Is \"West said she feels a custody order should be allowed to stand for the full term of the PFA - up to 18 months - as it does in many other counties in the state\" a correct answer?",
            "answer": "yes"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "Fossils give clues about major geological events. Fossils can also give clues about past climates. Fossils of ocean animals on the top of a mountain? Ocean animals have been found on the Earths tallest mountain. Its hard to believe, but it is true. These fossils were found at the top of Mt. Everest. Mt. Everest is the highest mountain on Earth. These fossils showed that this entire area was once at the bottom of a sea. It can only mean that Mt. Everest was uplifted. In fact, the entire Himalaya mountain range was raised. It was forced up from the collision of two continents. Fossils of plants are found in Antarctica. Now, Antarctica is almost completely covered with ice. Plants do not grow in Antarctica. According to fossils, they once did. This means that Antarctica was once much warmer than it is now. These fossils tell us about Antarcticas past climate. ",
            "question": "Based on the previous passage, How were the Himalayas \"uplifted\"? ",
            "choice_question": "Is \"The collision of two continents\" a correct answer?",
            "answer": "yes",
        },
        {
            "passage": "Fossils give clues about major geological events. Fossils can also give clues about past climates. Fossils of ocean animals on the top of a mountain? Ocean animals have been found on the Earths tallest mountain. Its hard to believe, but it is true. These fossils were found at the top of Mt. Everest. Mt. Everest is the highest mountain on Earth. These fossils showed that this entire area was once at the bottom of a sea. It can only mean that Mt. Everest was uplifted. In fact, the entire Himalaya mountain range was raised. It was forced up from the collision of two continents. Fossils of plants are found in Antarctica. Now, Antarctica is almost completely covered with ice. Plants do not grow in Antarctica. According to fossils, they once did. This means that Antarctica was once much warmer than it is now. These fossils tell us about Antarcticas past climate. ",
            "question": "Based on the previous passage, How were the Himalayas \"uplifted\"? ",
            "choice_question": "Is \"Magnetic forces\" a correct answer?",
            "answer": "no",
        },
        {
            "passage": "If you beat a dog in Schuylkill County, you'll probably get a $100 fine. If you repeatedly beat a woman, you'll probably get the same fine. In 2001, county judges heard 98 Protection From Abuse cases, finding the defendant guilty in 48 percent of those cases, either after a hearing or through a technical violation or plea. Of those found guilty, the majority were ordered to pay court costs, plus a $100 fine. No defendants were ordered to pay more than a $250 fine for violating the court order. In 27 percent of the cases, the charges were dismissed or the defendant was found not guilty. In the rest of the cases, charges were withdrawn or the matter is not yet resolved. Sarah T. Casey, executive director of Schuylkill Women in Crisis, finds it disturbing that in most cases, the fine for violating a PFA is little more than the fine someone would get for cruelty and abuse toward an animal. \"In most of the counties surrounding Schuylkill County, the penalties given for indirect criminal contempt are much stiffer than those in Schuylkill County,\" Casey said. \"What kind of message are we sending those who repeatedly violate Protection From Abuse orders? That it's OK to abuse women in Schuylkill County, because you'll only get a slap on the wrist?\" Under state law, the minimum fine for contempt of a PFA is $100; the maximum fine is $1,000 and up to six months in jail. Like others who are familiar with how the county's legal system does and doesn't work for victims of domestic violence, Casey believes some changes are in order. Valerie West, a manager/attorney with Mid-Penn Legal Services, with offices in Pottsville and Reading, regularly handles domestic violence cases. She finds fault with the local requirement that a custody order must be established within 30 days after a PFA is filed. West said she feels a custody order should be allowed to stand for the full term of the PFA - up to 18 months - as it does in many other counties in the state. \"It places an undue burden on the plaintiff, in terms of cost, finding...",
            "question": "Based on the previous passage, What solution is West offering and how is it different for a plaintiff from what is already being practiced? ",
            "choice_question": "Is \"West said she feels a custody order should be allowed to stand for the full term of the PFA - up to 18 months - as it does in many other counties in the state\" a correct answer?",
            "answer": "yes",
        },
        {
            "passage": "Sam Farragut is a sociopathic business executive in Southern California who forces a team of advertising agency employees to embark on a dangerous dirtbike trip to the Baja California desert in order to compete for his business .  The men are Warren Summerfield , a suicidal middle-aged ad executive who has been fired from the agency ; the straightlaced Paul McIlvain who is inattentive to his wife , and brash art designer Maxon who feels suddenly trapped after his girlfriend announces she is pregnant .  There are numerous long sequences of motorcycle riding on desert backroads .  Summerfield has been having an affair with McIlvian's wife .  He has not told his wife that he was fired and is simply serving out his tenure at the agency while looking for a new position .  His wife is actually aware of the affair .  Farragut convinces the ad men to make the motorcycle journey on the pretext of looking for a location to shoot a commercial .  In reality , Farragut is reckless and looking to involve the men in spontaneous edgy adventure of his own manipulation .  After they leave , McIlvain's wife suspects that Summerfield is planning to kill himself for the insurance money , but she can not convince Summerfield's wife to instigate a search .  The four men travel deeper into Mexico on isolated dirt roads .  At one point Summerfield contemplates plunging off a cliff .  After being humiliated by a young American couple in a Baja bar , Farragut tracks them down on the beach while accompanied by Maxon . ",
            "question": "Based on the previous passage, Under what pretext does a sociopathic company executive organize the motorcycle trip? ",
            "choice_question": "Is \"For a getaway trip\" a correct answer?",
            "answer": "no"
        }
    ]),
]

##############################################################################################################################

def multirc_metric(preds_by_question, golds_by_question):
    assert len(preds_by_question) == len(golds_by_question)
    agreement_count = 0
    correct_count = 0
    predict_count = 0
    accuracy_count = 0
    total_count = 0
    for p_id in range(len(preds_by_question)):
        predicted_ans = [int(g.lower() == "yes") for g in preds_by_question[p_id]]
        gold_ans = [int(g.lower() == "yes") for g in golds_by_question[p_id]]
        assert len(predicted_ans) == len(gold_ans)
        total_count += len(predicted_ans)
        if all([p == g for p, g in zip(predicted_ans, gold_ans)]):
            accuracy_count += 1
        predict_count += sum(predicted_ans)
        correct_count += sum(gold_ans)
        agreement_count += sum([a * b for (a, b) in zip(gold_ans, predicted_ans)])

    p1 = (1.0 * agreement_count / predict_count) if predict_count > 0.0 else 1.0
    r1 = (1.0 * agreement_count / correct_count) if correct_count > 0.0 else 1.0
    acc = (1.0 * accuracy_count / total_count) if total_count > 0.0 else 1.0
    return {"precision": p1, "recall": r1, "f1a": 2 * r1 * p1 / (p1 + r1), "accuracy": acc}


class MultiRCDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def transform_df(self, df):
        """Group data by passage for accurate multiRC metrics."""
        by_passage_question_df = defaultdict(list)
        for _, row in df.iterrows():
            passage, question, choice_question = row["inputs_pretokenized"].split("\n")
            new_key = passage + "\n" + question
            by_passage_question_df[new_key].append({"choice_question": choice_question, "answer": row["targets_pretokenized"]})
        for_df = []
        for key, d in by_passage_question_df.items():
            passage, question = key.split("\n")
            for_df.append({"passage": passage, "question": question, "choice_questions": [x["choice_question"] for x in d], "answers": [x["answer"] for x in d]})
        return pd.DataFrame(for_df)

    def read_data(self, save_dir, overwrite_data):
        save_data = Path(f"{save_dir}/{self.task_name}/data.feather")
        if not save_data.exists() or overwrite_data:
            test_data = self.transform_df(pd.read_feather(f"{self.data_dir}/{self.val_split}.feather"))
            train_data = self.transform_df(pd.read_feather(f"{self.data_dir}/train.feather"))
            test_data.to_feather(f"{save_data}")
        else:
            print(f"Reading train data from {save_data}")
            train_data = self.transform_df(pd.read_feather(f"{self.data_dir}/train.feather"))
            test_data = pd.read_feather(f"{save_data}")

        print(f"Test Data Size: {len(test_data)}")
        print(f"Train Data Size: {len(train_data)}")
        return test_data, train_data


    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            answer_prompt_examples[boost_id],
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
        cum_ind = 0
        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            # Task requires multiple questions per passage for scoring
            all_q_preds = []
            all_q_golds = []
            fs_exs = []
            if do_few_shot:
                cnt = 0
                for s_ind, s_row in few_shot_df.iterrows():
                    passage = s_row["passage"]
                    question = s_row["question"]
                    for choice_q, gold in zip(s_row["choice_questions"], s_row["answers"]):
                        cnt += 1
                        fs_exs.append({
                            "passage": passage,
                            "question": question,
                            "choice_question": choice_q,
                            "answer": gold,
                        })
                        # Take one question per passage
                        break
            passage = row["passage"]
            question = row["question"]
            all_prompts = []
            for q_ind, (choice_q, gold) in enumerate(zip(row["choice_questions"], row["answers"])):
                exs = fs_exs[:]
                exs.append({
                    "passage": passage,
                    "question": question,
                    "choice_question": choice_q,
                    "answer": ""
                })
                pmp = answer_prompt(pd.DataFrame(exs))
                all_prompts.append(pmp)
                res = get_response(
                    pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=8,
                    stop_token="\n",
                )
                all_q_golds.append(gold.strip().lower())
                all_q_preds.append(res.strip().lower())
                # row as dict - convert ndarry to list
                row_as_dict = row.to_dict()
                for k, v in list(row_as_dict.items()):
                    if isinstance(v, np.ndarray):
                        row_as_dict[k] = v.tolist()
                entry = {
                    "ind": cum_ind,
                    "example_ind": ind,
                    "question_ind": q_ind,
                    "example": row_as_dict,
                    "base_prompt": pmp,
                    "pred": res.strip().lower(),
                    "gold": gold.strip().lower(),
                }
                expt_log[cum_ind] = entry
                cum_ind += 1
            if i == 0:
                print(pmp)
            preds.append(all_q_preds)
            golds.append(all_q_golds)

        report = multirc_metric(preds_by_question=preds, golds_by_question=golds)
        return expt_log, report["f1a"]

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest, overwrite_manifest)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest, overwrite_manifest)
        # This task has nested labels (i.e. shape # test examples X number sub questions X number boosts)
        # Flatten over first two dims for merging and then undo
        all_boost_preds_flattened = [pred for pred_q_and_boost in all_boost_preds for pred in pred_q_and_boost]
        all_boost_train_preds_flattened = [pred for pred_q_and_boost in all_boost_train_preds for pred in pred_q_and_boost]
        train_labels_flattened = [label for label_q in train_labels for label in label_q]
        all_boost_preds_flattened = np.array(all_boost_preds_flattened)
        all_boost_train_preds_flattened = np.array(all_boost_train_preds_flattened)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds_flattened, all_boost_train_preds_flattened, train_labels_flattened, expt_log, expt_log_train)
        preds_unflattened = []
        cum_i = 0
        for i in range(len(all_boost_preds)):
            preds_unflattened.append([preds[cum_i + j] for j in range(len(all_boost_preds[i]))])
            cum_i += len(all_boost_preds[i])
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0][0])):
            individual_accuracies.append(multirc_metric(preds_by_question=[[p[i] for p in pred_set] for pred_set in all_boost_preds], golds_by_question=labels)["f1a"])
        metric = multirc_metric(preds_by_question=preds_unflattened, golds_by_question=labels)["f1a"]
        return expt_log, expt_log_train, metric, individual_accuracies    
    
    def _run_decomp_single_data(
        self, test_data, boost_dfs, manifest, overwrite_manifest
    ):
        expt_log = {}
        all_boost_preds = []
        labels = []
        cum_ind = 0
        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            all_q_golds = []
            all_q_preds = []

            passage = row["passage"]
            question = row["question"]
            # Add prompt before the inner gold choice loop
            for q_ind, (choice_q, gold) in enumerate(zip(row["choice_questions"], row["answers"])):
                all_q_golds.append(gold.strip().lower())
                prompts_across_boost = []
                preds_across_boost = []
                for boost_examples in boost_dfs:
                    prompt_suffix = answer_prompt(boost_examples[0])
                    pmp = f"{prompt_suffix}\n\nPassage: {passage}\nQuestion: {question}\n{choice_q}\nAnswer:"
                    # Single list pmp for one step decomp
                    prompts_across_boost.append([pmp])
                    res = get_response(
                        pmp,
                        manifest,
                        overwrite=bool(overwrite_manifest),
                        max_toks=8,
                    )
                    preds_across_boost.append(res.split("\n\n")[0].strip().lower())
                all_q_preds.append(preds_across_boost)
                # row as dict - convert ndarry to list
                row_as_dict = row.to_dict()
                for k, v in list(row_as_dict.items()):
                    if isinstance(v, np.ndarray):
                        row_as_dict[k] = v.tolist()
                expt_log[cum_ind] = {
                    "ind": cum_ind,
                    "example_ind": ind,
                    "question_ind": q_ind,
                    "preds_boost": preds_across_boost,
                    "prompts": prompts_across_boost,
                    "example": row_as_dict,
                    "gold": gold.strip().lower(),
                }
                cum_ind += 1
            
            if i == 0:
                for pmp_set in prompts_across_boost:
                    print("\n".join(pmp_set))
            all_boost_preds.append(all_q_preds)
            labels.append(all_q_golds)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    task_name = "multirc"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_multirc_is_a_correct_answer_"
    decomp = MultiRCDecomp(task_name, data_dir, val_split="validation")
    decomp.run(args)


if __name__ == "__main__":
    main()
