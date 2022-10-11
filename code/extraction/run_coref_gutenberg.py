from allennlp.predictors.predictor import Predictor
import copy
import allennlp_models.tagging
import pickle
import json
import re
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pickle


srl_cache = {}
ner_cache = {}
global_chapter_ids = []


def get_char_idx_from_token_idx(tokens, token_idx):
    ret = 0
    for t in tokens[:token_idx]:
        ret += len(t)
    return ret


def get_token_from_char_span(s, char_start, char_end):
    return s[char_start:char_end]


tokenized_chapters = {}
for line in [x.strip() for x in open("gutenberg_coref_doc.jsonl")]:
    l = line.split("\t")[0]
    idx = int(line.split("\t")[1])
    if idx <= 400 or idx > 18000:
        continue
    tokens = json.loads(l)["document"]
    tokenized_chapters[idx] = tokens
    global_chapter_ids.append(idx)


def get_token_from_char_pair(chapter_id, char_start, char_end):
    tokens = tokenized_chapters[chapter_id]
    accum = 0
    ret = []
    for t in tokens:
        if accum >= char_start and accum <= char_end:
            ret.append(t)
        accum += len(t)
    return ret


srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
verb_list = set([x.strip() for x in open("verb_list.txt").readlines()])
lemma = WordNetLemmatizer()


def get_verb(tags, words):
    for i, t in enumerate(tags):
        if t == "B-V":
            return words[i]
    return ""


def get_arg0_index(tags):
    start = -1
    end = -1
    for i, t in enumerate(tags):
        if t == "B-ARG0":
            start = i
            end = i
            for j in range(i + 1, len(tags)):
                if "ARG0" in tags[j]:
                    end = j
                else:
                    break
    return start, end


def get_char_idx_of_speaker_from_srl(sentence, char_start_idx):
    if len(sentence.split()) > 300:
        return None, None
    sentence_key = sentence.replace(" ", "").lower()
    if sentence_key in srl_cache:
        obj = srl_cache[sentence_key]
        print("hit")
    else:
        print("missed")
        try:
            obj = srl_predictor.predict(sentence=sentence)
        except:
            return None, None
        srl_cache[sentence_key] = obj
    for verb in obj["verbs"]:
        verb_surface = get_verb(verb["tags"], obj["words"])
        verb_normalized = lemma.lemmatize(verb_surface.lower(), "v")
        if verb_normalized in verb_list:
            left_words = []
            right_words = []
            default_key = "ARG1"
            found = False
            for t in verb["tags"]:
                if default_key in t:
                    found = True
            if not found:
                default_key = "ARG2"
            for i, tag in enumerate(verb["tags"]):
                if "B-{}".format(default_key) == verb["tags"][i]:
                    if i > 1:
                        left_words.append(obj["words"][i - 2])
                    if i > 0:
                        left_words.append(obj["words"][i - 1])
                    left_words.append(obj["words"][i])
                    for k in range(i, len(verb["tags"])):
                        if default_key in verb["tags"][k] and (k == len(verb["tags"]) - 1 or default_key not in verb["tags"][k + 1]):
                            right_words.append(obj["words"][k])
                            if k < len(verb["tags"]) - 1:
                                right_words.append(obj["words"][k + 1])
                            if k < len(verb["tags"]) - 2:
                                right_words.append(obj["words"][k + 2])
                            break
            valid = False
            if '"' in left_words:
                if len(right_words) == 3:
                    if right_words[-1] == '"' and right_words[1] in string.punctuation:
                        valid = True
                    if right_words[0] == '"' or right_words[1] == '"':
                        valid = True
                else:
                    if '"' in right_words:
                        valid = True
            if valid:
                start_token_idx, end_token_idx = get_arg0_index(verb["tags"])
                if start_token_idx == -1 or end_token_idx == -1:
                    return None, None
                return char_start_idx + get_char_idx_from_token_idx(obj["words"], start_token_idx), char_start_idx + get_char_idx_from_token_idx(obj["words"], end_token_idx)
    return None, None


def get_ne_token_surface_from_sentences(sentences):
    ret_set = set()
    for sentence in sentences:
        ner_key = format_line_to_model(sentence).replace(" ", "").lower()
        if ner_key in ner_cache:
            ner_obj = ner_cache[ner_key]
        else:
            try:
                ner_obj = ner_predictor.predict(format_line_to_model(sentence))
            except:
                continue
            ner_cache[ner_key] = ner_obj
        for i in range(0, len(ner_obj['words'])):
            if "PER" in ner_obj['tags'][i]:
                ret_set.add(ner_obj['words'][i])
    return list(ret_set)


def get_ne_token_surface_inner_quotation(sentence):
    ner_key = format_line_to_model(sentence).replace(" ", "").lower()
    if len(format_line_to_model(sentence)) > 350:
        return []
    ret = []
    if ner_key in ner_cache:
        ner_obj = ner_cache[ner_key]
    else:
        try:
            ner_obj = ner_predictor.predict(format_line_to_model(sentence))
        except:
            return []
    accum = 0
    for i in range(0, len(ner_obj['words'])):
        if "PER" in ner_obj['tags'][i]:
            if check_in_quotes(sentence, accum, accum, 0):
                ret.append(ner_obj['words'][i])
        accum += len(ner_obj['words'][i])
    return ret


def format_line_to_model(line):
    return line.replace("``", '"').replace("''", '"').replace("_", "")


def get_speaker_name(print_set, regular_vote=False, ner_words=[]):
    for mention in print_set:
        tokens = mention.split()
        valid = True
        for t in tokens:
            if not t[0].isupper():
                valid = False
        valid_ner = False
        for t in tokens:
            if t in ner_words:
                valid_ner = True
        if valid and valid_ner:
            return mention
    return None


def get_name_prediction(vote, harsh_inference=True, threshold=None):
    ret_key = None
    ret_vote = -1000
    for key in vote:
        if vote[key] > ret_vote:
            ret_vote = vote[key]
            ret_key = key
    counter = 0
    for key in vote:
        if vote[key] == ret_vote:
            counter += 1
    if counter > 1 or ret_vote <= 0:
        return None
    if harsh_inference:
        if threshold is not None:
            if ret_vote < threshold:
                return None
        elif ret_vote < 2:
            return None
    return ret_key


def check_in_quotes(original_line, node_start, node_end, accumulator):
    node_start = node_start - accumulator
    node_end = node_end - accumulator
    line = original_line.replace('``', "^").replace("''", "@").replace("_", "").replace(" ", "")
    assert node_start >= 0
    assert node_end < len(line)
    valid = True
    for i in range(node_start, -1, -1):
        if line[i] == '@':
            valid = False
        if line[i] == '^':
            break
    for i in range(node_end, len(line)):
        if line[i] == '@':
            break
        if line[i] == '^':
            valid = False
    return valid


def check_is_quote(line):
    return "``" in line and "''" in line


def get_sentence_surface_from_char_span(chapter, char_start, char_end):
    line_accumulation = 0
    for i, sent in enumerate(chapter):
        start = line_accumulation
        end = start + len(format_line_to_model(sent).replace(" ", ""))
        line_accumulation = end
        if char_start >= start and char_end < end:
            return sent
    return None


def get_cluster_owner(chapter_idx, chapter, node_start, node_end, resolved_quote_prediction_map):
    direction = 0
    surface = " ".join(get_token_from_char_pair(chapter_idx, node_start, node_end))
    if surface in ["i", "me", "my", "myself"]:
        direction = 1
    if surface in ["you", "yours", "you", "she", "her", "hers", "he", "his", "him", "himself", "herself", "my dear"]:
        direction = -1
    sentence = get_sentence_surface_from_char_span(chapter, node_start, node_end)
    if sentence is not None:
        if sentence in resolved_quote_prediction_map:
            return direction, resolved_quote_prediction_map[sentence]
    return direction, None


def cluster_sanity_check(chapter_idx, cluster):
    valid = True
    surfaces = []
    for node in cluster:
        surfaces.append(" ".join(get_token_from_char_pair(chapter_idx, node[0], node[1])).lower())
    self_counter = 0
    other_counter = 0
    for surface in surfaces:
        if surface in ["i", "me", "my", "myself"]:
            self_counter += 1
        if surface in ["you", "yours", "you", "she", "her", "hers", "he", "his", "him", "himself", "herself", "my dear"]:
            other_counter += 1
    if self_counter > 0 and other_counter > 0:
        valid = False
    for node_a in cluster:
        for node_b in cluster:
            if node_a[0] == node_b[0] and node_a[1] < node_b[1]:
                valid = False
    return valid


def parse_overlap(soft_mode=False, coref_path="gutenberg_coref_3sent.jsonl", harsh_inference=True, input_prediction_map=None, threshold=None, regular_vote=False):
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
        registered_lines.append(l)
    quote_prediction_map = {}
    total_segments = 0
    for l in [x.strip() for x in open(coref_path).readlines()]:
        json_content, chapter_idx, sent_start, sent_end, _ = l.split("\t")
        chapter_idx = int(chapter_idx)
        if chapter_idx in global_chapter_ids:
            total_segments += 1
    print("Total segments: {}".format(str(total_segments)))
    counter = 0
    for l in [x.strip() for x in open(coref_path).readlines()]:
        json_content, chapter_idx, sent_start, sent_end, _ = l.split("\t")
        chapter_idx = int(chapter_idx)
        if chapter_idx not in global_chapter_ids:
            continue
        counter += 1
        print("Processing segment {}".format(str(counter)))
        obj = json.loads(json_content)
        char_idx_accumulation = 0
        for sent in all_chapters[chapter_idx][:int(sent_start)]:
            char_idx_accumulation += len(sent.replace(" ", "").replace("``", '"').replace("''", '"').replace("_", ""))
        clusters = obj['clusters']
        tokens = obj['document']
        if tokens[0][0].isalpha():
            if tokens[0][0].lower() != all_chapters[chapter_idx][int(sent_start)][0].lower():
                print(tokens)
                print(all_chapters[chapter_idx][int(sent_start)])
        current_predicted_cluster = []
        for cluster in clusters:
            group = []
            for node_start, node_end in cluster:
                group.append(
                    (get_char_idx_from_token_idx(tokens, node_start) + char_idx_accumulation,
                     get_char_idx_from_token_idx(tokens, node_end) + char_idx_accumulation
                     )
                )
            current_predicted_cluster.append(group)
        line_accumulator = char_idx_accumulation
        ner_lines = []
        for line in all_chapters[chapter_idx][int(sent_start):int(sent_end)]:
            ner_lines.append(format_line_to_model(line))
        ner_words = get_ne_token_surface_from_sentences(ner_lines)
        segment = " ".join(all_chapters[chapter_idx][int(sent_start):int(sent_end)])
        segment = segment.replace("``", '"').replace("''", '"').replace("_", "")
        for line in all_chapters[chapter_idx][int(sent_start):int(sent_end)]:
            if chapter_idx not in global_chapter_ids:
                continue
            char_start_idx = line_accumulator
            clean_line = re.sub('\s+', '', format_line_to_model(line))
            char_end_idx = line_accumulator + len(clean_line)
            speaker_char_start, speaker_char_end = get_char_idx_of_speaker_from_srl(format_line_to_model(line), line_accumulator)
            if line.startswith("``") and line.endswith("''"):
                # Current line is a quotation
                votes = {}
                if speaker_char_start is not None:
                    speaker_surface = get_token_from_char_pair(chapter_idx, speaker_char_start, speaker_char_end)
                    speaker_person = get_speaker_name({" ".join(speaker_surface)}, ner_words=ner_words)
                    if speaker_person is not None:
                        if speaker_person not in votes:
                            votes[speaker_person] = 0
                        votes[speaker_person] += 5
                inner_names = get_ne_token_surface_inner_quotation(line)
                for surface in inner_names:
                    tmp_max_name = get_speaker_name({surface}, ner_words=ner_words)
                    if tmp_max_name is not None:
                        if tmp_max_name not in votes:
                            print("vote set to 0 for {} because inner name reference".format(tmp_max_name))
                            votes[tmp_max_name] = 0
                        print("vote -1 for {} because inner name reference".format(tmp_max_name))
                        votes[tmp_max_name] -= 1
                for cluster in current_predicted_cluster:
                    if not cluster_sanity_check(chapter_idx, cluster) and harsh_inference:
                        continue
                    if not harsh_inference:
                        regular_vote = True
                    for node in cluster:
                        if speaker_char_start is not None:
                            if node[0] == speaker_char_start and node[1] == speaker_char_end:
                                print_set = set()
                                for n in cluster:
                                    print_set.add(" ".join(get_token_from_char_pair(chapter_idx, n[0], n[1])))
                                vote_name = get_speaker_name(print_set, regular_vote, ner_words=ner_words)
                                if vote_name is not None:
                                    if vote_name not in votes:
                                        print("vote set to 0 for {} because direct speaker coreference, node {}".format(
                                            vote_name, str(node)))
                                        votes[vote_name] = 0
                                    print("vote +1 for {} because direct speaker coreference, node {}".format(vote_name, str(node)))
                                    votes[vote_name] += 1

                        if node[0] >= char_start_idx and node[1] < char_end_idx and check_in_quotes(line, node[0], node[1], line_accumulator):
                            print_set = set()
                            for n in cluster:
                                print_set.add(" ".join(get_token_from_char_pair(chapter_idx, n[0], n[1])))
                            surface = " ".join(get_token_from_char_pair(chapter_idx, node[0], node[1])).lower()
                            vote_name = get_speaker_name(print_set, regular_vote, ner_words=ner_words)
                            if vote_name is not None:
                                if vote_name not in votes:
                                    print("vote set to 0 for {} because self reference, node {}, accu {}".format(vote_name, str(node), str(line_accumulator)))
                                    votes[vote_name] = 0
                                if surface in ["i", "me", "my", "myself"]:
                                    print("vote +1 for {} because self reference, node {}, accu {}".format(vote_name, str(node), str(line_accumulator)))
                                    votes[vote_name] += 1
                                if surface in ["you", "yours", "you", "she", "her", "hers", "he", "his", "him", "himself", "herself", "my dear"]:
                                    print("vote -1 for {} because self reference, node {}, accu {}".format(vote_name, str(node), str(line_accumulator)))
                                    votes[vote_name] -= 1
                if line not in quote_prediction_map:
                    quote_prediction_map[line] = []
                quote_prediction_map[line].append(votes)
                print("------first round---------")
                print(segment)
                print(line)
                print(ner_words)
                print(votes)
            line_accumulator += len(clean_line)

    if input_prediction_map is None:
        resolved_quote_prediction_map = {}
    else:
        resolved_quote_prediction_map = input_prediction_map
    for key in quote_prediction_map:
        if key in resolved_quote_prediction_map:
            continue
        combined_vote = {}
        votes = quote_prediction_map[key]
        for vote in votes:
            max_name = get_name_prediction(vote, harsh_inference=False)
            if max_name is not None:
                if max_name not in combined_vote:
                    combined_vote[max_name] = 0
                combined_vote[max_name] += 1
        pred = "Unknown"
        max_name = get_name_prediction(combined_vote, harsh_inference=harsh_inference, threshold=threshold)
        if max_name is not None:
            valid = True
            for vote in votes:
                if max_name in vote and vote[max_name] < 0:
                    valid = False
            if valid:
                pred = max_name
                resolved_quote_prediction_map[key] = max_name
            if pred == "Unknown" and soft_mode:
                resolved_quote_prediction_map[key] = max_name
            print("----------")
            print(key)
            print(max_name)
            print(combined_vote[max_name])
    print("this round: parse_overlap() with harsh_inference={} and coref_path={}".format(str(harsh_inference), str(coref_path)))
    return all_chapters, resolved_quote_prediction_map


all_chapters, resolved_quote_prediction_map = parse_overlap(harsh_inference=True)

with open("all_predictions_gutenberg.pkl", "wb") as f_out_1:
    pickle.dump(resolved_quote_prediction_map, f_out_1)
