import os
import spacy
import json
import copy
import re
from allennlp.predictors.predictor import Predictor
import pickle
import random

nlp = spacy.load("en_core_web_sm")

def check_is_quote(line):
    return "``" in line and "''" in line


def format_line(line):
    return line.replace("``", '"').replace("''", '"').replace("_", "")


def get_quote_counter(sentence):
    quote_counter = 0
    for char in sentence:
        if char == '"':
            quote_counter += 1
    return quote_counter


def run_gutenberg_coref():
    novel_lines = [x.strip() for x in open("gutenberg_pp_style_sample.txt").readlines()]
    all_chapters = []
    registered_lines = []
    for l in novel_lines:
        if l == "":
            continue
        if l.startswith("CHAPTER") and len(registered_lines) > 0:
            all_chapters.append(copy.deepcopy(registered_lines))
            registered_lines = []
            continue
        if l.startswith("CHAPTER"):
            continue
        registered_lines.append(re.sub('\s+', ' ', l.replace("``", '"').replace("''", '"').replace("_", "")))
    print("Total chapters {}".format(str(len(all_chapters))))

    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
    f_out = open("gutenberg_coref_doc.jsonl", "w")
    f_out_3sent = open("gutenberg_coref_3sent.jsonl", "w")
    for i, chapter in enumerate(all_chapters):
        if i < 400:
            continue
        content = " ".join(chapter)
        if len(content.split()) > 5000:
            continue
        r = predictor.predict(
            document=content
        )
        f_out.write("\t".join([json.dumps(r), str(i), str(0), str(0), " "]) + "\n")
        for j in range(0, len(chapter) - 3):
            right_bound = min(j+3, len(chapter))
            if right_bound - j < 1:
                continue
            content = " ".join(chapter[j:right_bound])
            if len(content.replace(" ", "")) < 1 or get_quote_counter(content) % 2 == 1 or get_quote_counter(content) == 0:
                continue
            r = predictor.predict(
                document=content
            )
            f_out_3sent.write("\t".join([json.dumps(r), str(i), str(j), str(right_bound), content]) + "\n")
        print("Finished chapter {}".format(str(i)))


ner_cache = {}
ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")


def get_mentions(sentence):
    ner_key = format_line(sentence).replace(" ", "").lower()
    if ner_key in ner_cache:
        r = ner_cache[ner_key]
    else:
        try:
            r = ner_predictor.predict(format_line(sentence))
            ner_cache[ner_key] = r
        except:
            return []
    tokens = r["words"]
    mentions = []
    for i, t in enumerate(tokens):
        if t[0].isupper():
            right_bound = i + 1
            for j in range(i+1, len(tokens)):
                if tokens[j][0].isupper():
                    right_bound = j + 1
                else:
                    break
            valid = False
            for k in range(i, right_bound):
                if "PER" in r["tags"][k]:
                    valid = True
            if valid:
                mentions.append(" ".join(tokens[i:right_bound]))
    return mentions


def check_in_quotes(original_line, node_start, node_end):
    line = original_line
    assert node_start >= 0
    valid = True
    for i in range(node_start, 1, -1):
        if line[i-2:i] == "''":
            valid = False
        if line[i-2:i] == '``':
            break
    for i in range(node_end, len(line)):
        if line[i-2:i] == "''":
            break
        if line[i-2:i] == '``':
            valid = False
    return valid


def format_to_roberta(output_path):
    with open("", "rb") as f_in:
        prediction_map = pickle.load(f_in)

    print(len(prediction_map))
    novel_lines = [x.strip() for x in open("gutenberg_pp_style_sample.txt").readlines()]
    all_chapters = []
    registered_lines = []
    f_out = open(output_path, "w")
    for l in novel_lines:
        if l == "":
            continue
        if (l.startswith("CHAPTER") or l == "--Pride and Prejudice--") and len(registered_lines) > 0:
            all_chapters.append(copy.deepcopy(registered_lines))
            registered_lines = []
            continue
        if l.startswith("CHAPTER"):
            continue
        registered_lines.append(l)
    for cid, chapter in enumerate(all_chapters):
        if cid > 18000 or cid < 15000:
            continue
        for i, sentence in enumerate(chapter):
            if not check_is_quote(sentence):
                continue

            left_bound = random.choice([i-3, i-4])
            left_bound = max(0, left_bound)
            right_bound = random.choice([i+2, i+3])
            right_bound = min(len(chapter), right_bound)
            if sentence not in prediction_map:
                continue
            predicted_speaker_surface = prediction_map[sentence]
            mutate = False
            for sub_sentence in chapter[left_bound:right_bound]:
                if sub_sentence == sentence:
                    continue
                if predicted_speaker_surface.lower() in sub_sentence.lower():
                    mutate = True
            if random.random() < 1:
                mutate = False
            if predicted_speaker_surface not in sentence:
                mutate = False
            orig_sentence = sentence
            if mutate:
                lower_cased_sentence = sentence.lower()
                predicted_speaker_surface_lower = predicted_speaker_surface.lower()
                replace_positions = []
                for char_i in range(0, len(lower_cased_sentence)):
                    if lower_cased_sentence[char_i:char_i+len(predicted_speaker_surface_lower)] == predicted_speaker_surface_lower:
                        if not check_in_quotes(sentence, char_i, char_i+len(predicted_speaker_surface_lower)):
                            replace_positions.append(char_i)
                for p in replace_positions:
                    sentence = sentence[:p] + "someone" + sentence[p+len(predicted_speaker_surface_lower):]

            context = ""
            for sub_sentence in chapter[left_bound:i]:
                context += format_line(sub_sentence) + " </s> "
            context += format_line(sentence) + " </s> "
            for sub_sentence in chapter[i+1:right_bound]:
                context += format_line(sub_sentence) + " </s> "
            context = context.strip()

            if len(context.split()) > 400:
                continue
            if orig_sentence not in prediction_map:
                continue
            else:
                label = prediction_map[orig_sentence]
                res = re.findall(r'``(.*?)\'\'', sentence)
                quote_marker = "[X]".join(res)
                question = 'who said "{}"?'.format(quote_marker.replace("[X]", " "))
                all_mentions = []
                for s in chapter[left_bound:right_bound]:
                    all_mentions += get_mentions(s)
                all_mentions = list(set(all_mentions))
                random.shuffle(all_mentions)
                random_mapping = {}
                counter = 0
                naming_sequence = ["Person B", "Person C", "Person D", "Person E", "Person F", "Person G", "Person H", "Person K", "Person J", "Person L", "Person M", "Person N", "Person P", "Person Q", "Person T", "Person X", "Person Y", "Person Z"]
                random.shuffle(naming_sequence)
                for mm in all_mentions:
                    if mm not in random_mapping:
                        random_mapping[mm] = naming_sequence[counter]
                        counter += 1
                        if counter > 17:
                            break
                random_mapping_keys = list(random_mapping.keys())
                random_mapping_keys = sorted(random_mapping_keys, key=len, reverse=True)
                for key in random_mapping_keys:
                    context = context.replace(key, random_mapping[key])
                    question = question.replace(key, random_mapping[key])
                if label not in all_mentions:
                    continue
                if label not in random_mapping:
                    continue
                poi = random_mapping[label]
                involved_keys = set()
                for k in random_mapping:
                    involved_keys.add(random_mapping[k])
                involved_keys = list(involved_keys)
                random.shuffle(involved_keys)
                involved_keys_str = "People: " + "; ".join(involved_keys)
                if poi not in context:
                    continue
                if len(involved_keys) <= 1:
                    continue
                for char_k in range(0, len(involved_keys_str)):
                    if involved_keys_str[char_k:char_k+len(poi)] == poi:
                        f_out.write(involved_keys_str + " </s> " + context + "\t" + question + "\t" + str(char_k) + "\t" + str(char_k + len(poi)) + "\n")
                        f_out.flush()
