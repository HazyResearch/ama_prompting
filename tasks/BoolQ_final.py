#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

extract = InputOutputPrompt(
    input_formatter=lambda x: f"Context: {x['context']}\n\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["context", "question", "answer"],
    input_output_sep="\n\n",
    example_sep="\n\n----\n\n",
    instruction="Answer the question using the context.\n\n"
)


extract_examples = [
    pd.DataFrame([
        {
            "context": "Tonic water -- Tonic water (or Indian tonic water) is a carbonated soft drink in which quinine is dissolved. Originally used as a prophylactic against malaria, tonic water usually now has a significantly lower quinine content and is consumed for its distinctive bitter flavor. It is often used in mixed drinks, particularly in gin and tonic.",
            "question": "does tonic water still have quinine in it?",
            "answer": "yes"
        },
        {
            "context": "Northern bobwhite -- The northern bobwhite, Virginia quail or (in its home range) bobwhite quail (Colinus virginianus) is a ground-dwelling bird native to the United States, Mexico, and the Caribbean. It is a member of the group of species known as New World quails (Odontophoridae). They were initially placed with the Old World quails in the pheasant family (Phasianidae), but are not particularly closely related. The name ``bobwhite'' derives from its characteristic whistling call. Despite its secretive nature, the northern bobwhite is one of the most familiar quails in eastern North America because it is frequently the only quail in its range. Habitat degradation has likely contributed to the northern bobwhite population in eastern North America declining by roughly 85% from 1966-2014. This population decline is apparently range-wide and continuing.",
            "question": "is a quail the same as a bobwhite?",
            "answer": "yes"
        },
        {
            "context": "United States Department of Homeland Security -- In fiscal year 2017, it was allocated a net discretionary budget of $40.6 billion. With more than 240,000 employees, DHS is the third largest Cabinet department, after the Departments of Defense and Veterans Affairs. Homeland security policy is coordinated at the White House by the Homeland Security Council. Other agencies with significant homeland security responsibilities include the Departments of Health and Human Services, Justice, and Energy",
            "question": "is department of homeland security part of dod?",
            "answer": "no"
        }
    ]),
    pd.DataFrame([
        {
            "context": "Debit card cashback -- The services are restricted to debit cards where the merchant pays a fixed fee for the transaction, it is not offered on payments by credit card because they would pay a percentage commission on the additional cash amount to their bank or merchant service provider.",
            "question": "can i ask for cashback with a credit card?",
            "answer": "no"
        },
        {
            "context": "Bundle branch block -- A bundle branch block can be diagnosed when the duration of the QRS complex on the ECG exceeds 120 ms. A right bundle branch block typically causes prolongation of the last part of the QRS complex, and may shift the heart's electrical axis slightly to the right. The ECG will show a terminal R wave in lead V1 and a slurred S wave in lead I. Left bundle branch block widens the entire QRS, and in most cases shifts the heart's electrical axis to the left. The ECG will show a QS or rS complex in lead V1 and a monophasic R wave in lead I. Another normal finding with bundle branch block is appropriate T wave discordance. In other words, the T wave will be deflected opposite the terminal deflection of the QRS complex. Bundle branch block, especially left bundle branch block, can lead to cardiac dyssynchrony. The simultaneous occurrence of left and right bundle branch block leads to total AV block.",
            "question": "can you have a right and left bundle branch block?",
            "answer": "yes"
        },
        {
            "context": "Windsor Castle -- Queen Victoria and Prince Albert made Windsor Castle their principal royal residence, despite Victoria complaining early in her reign that the castle was ``dull and tiresome'' and ``prison-like'', and preferring Osborne and Balmoral as holiday residences. The growth of the British Empire and Victoria's close dynastic ties to Europe made Windsor the hub for many diplomatic and state visits, assisted by the new railways and steamships of the period. Indeed, it has been argued that Windsor reached its social peak during the Victorian era, seeing the introduction of invitations to numerous prominent figures to ``dine and sleep'' at the castle. Victoria took a close interest in the details of how Windsor Castle was run, including the minutiae of the social events. Few visitors found these occasions comfortable, both due to the design of the castle and the excessive royal formality. Prince Albert died in the Blue Room at Windsor Castle in 1861 and was buried in the Royal Mausoleum built at nearby Frogmore, within the Home Park. The prince's rooms were maintained exactly as they had been at the moment of his death and Victoria kept the castle in a state of mourning for many years, becoming known as the ``Widow of Windsor'', a phrase popularised in the famous poem by Rudyard Kipling. The Queen shunned the use of Buckingham Palace after Albert's death and instead used Windsor Castle as her residence when conducting official business near London. Towards the end of her reign, plays, operas, and other entertainments slowly began to be held at the castle again, accommodating both the Queen's desire for entertainment and her reluctance to be seen in public.",
            "question": "is buckingham palace the same as windsor castle?",
            "answer": "no"
        }
    ]),
    pd.DataFrame([
        {
            "context": "The Princess and the Goblin (film) -- The Princess and the Goblin (Hungarian: A hercegnő és a kobold) is a 1991 British-Hungarian-American animated musical fantasy film directed by József Gémes and written by Robin Lyons, an adaptation of George MacDonald's 1872 novel of the same name.",
            "question": "is the princess and the goblin a disney movie?",
            "answer": "no"
        },
        {
            "context": "Field marshal (United Kingdom) -- Field Marshal has been the highest rank in the British Army since 1736. A five-star rank with NATO code OF-10, it is equivalent to an Admiral of the Fleet in the Royal Navy or a Marshal of the Royal Air Force in the Royal Air Force (RAF). A Field Marshal's insignia consists of two crossed batons surrounded by yellow leaves below St Edward's Crown. Like Marshals of the RAF and Admirals of the Fleet, Field Marshals traditionally remain officers for life, though on half-pay when not in an appointment. The rank has been used sporadically throughout its history and was vacant during parts of the 18th and 19th centuries (when all former holders of the rank were deceased). After the Second World War, it became standard practice to appoint the Chief of the Imperial General Staff (later renamed Chief of the General Staff) to the rank on his last day in the post. Army officers occupying the post of Chief of the Defence Staff, the professional head of all the British Armed Forces, were usually promoted to the rank upon their appointment.",
            "question": "is there a field marshal in the british army?",
            "answer": "yes"
        },
        {
            "context": "Washington, D.C. -- The signing of the Residence Act on July 16, 1790, approved the creation of a capital district located along the Potomac River on the country's East Coast. The U.S. Constitution provided for a federal district under the exclusive jurisdiction of the Congress and the District is therefore not a part of any state. The states of Maryland and Virginia each donated land to form the federal district, which included the pre-existing settlements of Georgetown and Alexandria. Named in honor of President George Washington, the City of Washington was founded in 1791 to serve as the new national capital. In 1846, Congress returned the land originally ceded by Virginia; in 1871, it created a single municipal government for the remaining portion of the District.",
            "question": "is washington dc a part of a state?",
            "answer": "no"
        }
    ]),
    pd.DataFrame([
        {
            "context": "Legal issues in airsoft -- Under federal law, airsoft guns are not classified as firearms and are legal for all ages. People under the age of 18 are not permitted to buy airsoft guns over the counter in stores. However, a person of any age may use one (with the permission of their parents, of course, for anyone under 18). This is also the case for the laws in each state. However, in some major cities, the definition of a firearm within their respected ordinances includes propulsion by spring or compressed air, thus making airsoft subject to applicable laws. For example, airsoft guns within the state of California can only be bought by a person above the age of 18. However, no laws indicate an age requirement to sell airsoft guns. Generally speaking, toy, look-alike, and imitation firearms must have an orange tip during shipping and transportation.",
            "question": "do you have to be 18 to buy airsoft guns?",
            "answer": "yes"
        },
        {
            "context": "India national football team -- India has never participated in the FIFA World Cup, although the team did qualify by default for the 1950 World Cup after all the other nations in their qualification group withdrew. However, India withdrew prior to the beginning of the tournament. The team has also appeared three times in the Asia's top football competition, the AFC Asian Cup. Their best result in the competition occurred in 1964 when the team finished as runners-up. India also participate in the SAFF Championship, the top regional football competition in South Asia. They have won the tournament six times since it began in 1993. \nQuestion: has india ever played in fifa world cup.",
            "question": "has india ever played in fifa world cup?",
            "answer": "no"
        },
        {
            "context": "Pan-American Highway -- The Pan-American Highway is a network of roads measuring about 30,000 kilometres (19,000 mi) in total length. Except for a rainforest break of approximately 160 km (100 mi), called the Darién Gap, the road links almost all of the mainland countries of the Americas in a connected highway system. According to Guinness World Records, the Pan-American Highway is the world's longest ``motorable road''. However, because of the Darién Gap, it is not possible to cross between South America and Central America, alternatively being able to circumnavigate this terrestrial stretch by sea.",
            "question": "could you drive from north america to south america?",
            "answer": "no"
        }
    ]),
    pd.DataFrame([
        {
            "context": "Appointment and confirmation to the Supreme Court of the United States -- The appointment and confirmation of Justices to the Supreme Court of the United States involves several steps set forth by the United States Constitution, which have been further refined and developed by decades of tradition. Candidates are nominated by the President of the United States and must face a series of hearings in which both the nominee and other witnesses make statements and answer questions before the Senate Judiciary Committee, which can vote to send the nomination to the full United States Senate. Confirmation by the Senate allows the President to formally appoint the candidate to the court.",
            "question": "do supreme court justices have to be approved by congress?",
            "answer": "no"
        },
        {
            "context": "Glowplug -- Diesel engines, unlike gasoline engines, do not use spark plugs to induce combustion. Instead, they rely solely on compression to raise the temperature of the air to a point where the diesel combusts spontaneously when introduced to the hot high pressure air. The high pressure and spray pattern of the diesel ensures a controlled, complete burn. The piston rises, compressing the air in the cylinder; this causes the air's temperature to rise. By the time the piston reaches the top of its travel path, the temperature in the cylinder is very high. The fuel mist is then sprayed into the cylinder; it instantly combusts, forcing the piston downwards, thus generating power. The pressure required to heat the air to that temperature, however, requires a large and strong engine block.",
            "question": "is there a spark plug in diesel engine?",
            "answer": "no"
        },
        {
            "context": "Buffy the Vampire Slayer Season Eight -- Buffy the Vampire Slayer Season Eight is a comic book series published by Dark Horse Comics from 2007 to 2011. The series serves as a canonical continuation of the television series Buffy the Vampire Slayer, and follows the events of that show's final televised season. It is produced by Joss Whedon, who wrote or co-wrote three of the series arcs and several one-shot stories. The series was followed by Season Nine in 2011.",
            "question": "is there a season 8 of buffy the vampire slayer?",
            "answer": "yes"
        }
    ]),
    pd.DataFrame([
        {
            "context": "Uterus -- The uterus (from Latin ``uterus'', plural uteri) or womb is a major female hormone-responsive secondary sex organ of the reproductive system in humans and most other mammals. In the human, the lower end of the uterus, the cervix, opens into the vagina, while the upper end, the fundus, is connected to the fallopian tubes. It is within the uterus that the fetus develops during gestation. In the human embryo, the uterus develops from the paramesonephric ducts which fuse into the single organ known as a simplex uterus. The uterus has different forms in many other animals and in some it exists as two separate uteri known as a duplex uterus.",
            "question": "are the womb and the uterus the same thing?",
            "answer": "yes"
        },
        {
            "context": "Super Bowl XLVII -- Super Bowl XLVII was an American football game between the American Football Conference (AFC) champion Baltimore Ravens and the National Football Conference (NFC) champion San Francisco 49ers to decide the National Football League (NFL) champion for the 2012 season. The Ravens defeated the 49ers by the score of 34--31, handing the 49ers their first Super Bowl loss in franchise history. The game was played on Sunday, February 3, 2013 at Mercedes-Benz Superdome in New Orleans, Louisiana. This was the tenth Super Bowl to be played in New Orleans, equaling Miami's record of ten in an individual city.",
            "question": "did the 49ers win the superbowl in 2012?",
            "answer": "no"
        },
        {
            "context": "Blacklight -- A blacklight (or often black light), also referred to as a UV-A light, Wood's lamp, or simply ultraviolet light, is a lamp that emits long-wave (UV-A) ultraviolet light and not much visible light.",
            "question": "are black lights and uv lights the same thing?",
            "answer": "yes"
        }
    ]),
    pd.DataFrame([
        {
            "context": "2018 Winter Olympics -- In June 2017, Ubisoft announced that it would release an expansion pack for its winter sports video game Steep entitled Road to the Olympics, which features new game modes and content inspired by the 2018 Winter Olympics.",
            "question": "will there be a winter olympics video game?",
            "answer": "yes"
        },
        {
            "context": "Castor oil -- Castor oil is a vegetable oil obtained by pressing the seeds of the castor oil plant (Ricinus communis). The common name ``castor oil'', from which the plant gets its name, probably comes from its use as a replacement for castoreum, a perfume base made from the dried perineal glands of the beaver (castor in Latin).",
            "question": "is vegetable oil and castor oil the same?",
            "answer": "no"
        },
        {
            "context": "The Mother (How I Met Your Mother) -- Tracy McConnell, better known as ``The Mother'', is the title character from the CBS television sitcom How I Met Your Mother. The show, narrated by Future Ted, tells the story of how Ted Mosby met The Mother. Tracy McConnell appears in 8 episodes from ``Lucky Penny'' to ``The Time Travelers'' as an unseen character; she was first seen fully in ``Something New'' and was promoted to a main character in season 9. The Mother is played by Cristin Milioti.",
            "question": "does how i met your mother ever show ted's wife?",
            "answer": "yes"
        }
    ]),
]

prefix_select_zeroshot = """Answer the question."""


class BoolQDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)


    def get_boost_decomp_examples(self, train_data, boost_id):
        return [
            extract_examples[boost_id],
        ]

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = ['No', 'Yes']
        num_per_class = int(np.ceil(k_shot / len(labels)))
        print(f"Selecting {num_per_class} examples per class.")

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1
            sub_df = train_data[train_data["targets_pretokenized"] == label]
            sub_df = sub_df.sample(num_per_class)
            dfs.append(sub_df)
            total_in_context += num_per_class
            if total_in_context == k_shot:
                break
        mini_df = pd.concat(dfs)
        return mini_df

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
            text = row['inputs_pretokenized']
            passage = text.split("\nQuestion")[0].strip()
            question = text.split("\nQuestion")[-1].split("\nAnswer")[0].strip().strip(":").strip().strip("\n").strip("?").strip()
            gold = row['targets_pretokenized']

            icl_str = f"{prefix_select_zeroshot}"
            if do_few_shot:
                for s_ind, s_row in few_shot_df.iterrows():
                    s_text = s_row['inputs_pretokenized']
                    s_passage = s_text.split("\nQuestion")[0].strip("\n").strip()
                    s_question = s_text.split("\nQuestion")[-1].split("\nAnswer")[0].strip().strip(":").strip().strip("\n").strip("?").strip()
                    icl_str += f"\n\nContext: {s_passage}\nQuestion: {s_question}?\nAnswer: {s_row['targets_pretokenized']}"

            prompt = f"{icl_str}\n\nContext: {passage}\nQuestion: {question}?\nAnswer:"

            if i == 0:
                print(prompt)

            answer = get_response(
                prompt,
                manifest,
                overwrite=bool(overwrite_manifest),
                max_toks=10,
                stop_token="\n\n",
            )
            answer = answer.strip("\n").lower()
            pred = answer.strip()
            gold = gold.strip().lower()

            entry = {
                "ind": ind,
                "example": text,
                "base_prompt": prompt,
                "pred": pred,
                "gold": gold,
            }
            expt_log[ind] = entry

            preds.append(pred)
            labels.append(gold)

        report = classification_report(labels, preds, output_dict=True)
        return expt_log, report["accuracy"]

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


        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            text = row['inputs_pretokenized']
            passage = text.split("\nQuestion")[0].strip()
            question = text.split("\nQuestion")[-1].split("\nAnswer")[0].replace(
                "True or False?", "").strip().strip(":").strip().strip("?").strip()
            gold = row['targets_pretokenized']
            if i == run_limit:
                break
            
            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                all_prompts = []
                prompt_suffix = extract(boost_examples[0])
                extract_prompt = f"{prompt_suffix}\n\n----\n\nContext: {{passage:}}\n\nQuestion: {{question:}}?\n\nAnswer:" 
                extract_pmp = extract_prompt.format(passage=passage, question=question)
                output = get_response(
                    extract_pmp,
                    manifest,
                    overwrite=bool(overwrite_manifest),
                    max_toks=5,
                )
                all_prompts.append(extract_pmp)
                
                if i == 0:
                    print(extract_pmp)
                
                answer = output.strip("\n").lower()
                answer = [a for a in answer.split("\n") if a][0]
                answer = "".join(
                    [a for a in answer if a not in [".", ",", "?", ";", ":", "'", '"']]
                )
                gold = gold.strip().lower()
                pred = answer

                is_yes = "yes" in pred.split()
                is_no = "no" in pred.split()
                pred = "No"
                if is_yes and (not is_no):
                    pred = "Yes"
                if is_no and (not is_yes):
                    pred = "No"
                pred = pred.lower()
                
                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)

            entry = {
                "ind": ind,
                "example": text,
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
    args.num_boost = 5
    task_name = "super_glue_boolq"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_boolq_GPT_3_Style/"
    boolq = BoolQDecomp(task_name, data_dir)
    boolq.run(args)


if __name__ == "__main__":
    main()
