from tensorflow.keras.utils import to_categorical
from config import TOKENIZER, MAX_LEN, NUM_TAGS
from tqdm import tqdm
import numpy as np


def join_subtokens(ids, labels):
    joined_text = []
    joined_tags = []
    for ids_row, labels_row in zip(ids, labels):
        joined_text_row = []
        joined_tags_row = []
        for sub_token, label in zip(ids_row, labels_row):
            if sub_token.startswith("##"):
                joined_text_row[-1] = joined_text_row[-1] + sub_token[2:]
            else:
                joined_text_row.append(sub_token)
                joined_tags_row.append(label)

        joined_text.append(joined_text_row)
        joined_tags.append(joined_tags_row)

    return joined_text, joined_tags


def encode(sentence, tag, tag2idx):
    sent_enc = []
    tag_enc = []
    for word, label in zip(sentence, tag):
        word_enc = TOKENIZER.encode(word, add_special_tokens=False)
        token_len = len(word_enc)
        sent_enc.extend(word_enc)
        tag_enc.extend([tag2idx[label]]*token_len)
    sent_enc = sent_enc[:MAX_LEN-2]
    tag_enc = tag_enc[:MAX_LEN-2]
    sent_enc = [101] + sent_enc + [102]
    tag_enc = [0] + tag_enc + [0] 
    num = MAX_LEN - len(sent_enc)
    sent_enc += [0]*num
    tag_enc += [0]*num

    return sent_enc, tag_enc


def prepare_data(text, tags, tag2idx):
    text_encoded = []
    tags_encoded = []

    for sentence, labels in tqdm(zip(text, tags), total=len(text)):
        sent_enc, tag_enc = encode(sentence, labels, tag2idx)
        text_encoded.append(sent_enc)
        tags_encoded.append(tag_enc)

    mask = [[int(i!=0) for i in ii] for ii in text_encoded]
    labels = np.array([to_categorical(i, NUM_TAGS) for i in tags_encoded])
    text_encoded = np.array(text_encoded)
    mask = np.array(mask)

    return text_encoded, mask, labels


def expand_entity_dict(entity_dict, ids, labels, threshold):
    new_dict = {}

    text_joined = []
    tags_joined = []
  
    for p,q in zip(ids, labels):
        text_joined.extend(p)
        tags_joined.extend(q)

    for entity, label in zip(text_joined, tags_joined):
        if entity in new_dict:
            new_dict[entity][1] += 1
        else:
            new_dict[entity] = [label, 1]

    for k,v in new_dict.items():
        if v[1] > threshold and k not in entity_dict:
            entity_dict[k] = v[0]

    return entity_dict


# def label_encoding(entity_dict):
#     entities = list(set(entity_dict.values()))
#     tag2idx = {v:i+1 for i,v in enumerate(entities)}
#     tag2idx['PAD'] = 0
#     idx2tag = {i:v for v,i in tag2idx.items()}
#     return tag2idx, idx2tag


def label_encoding(labels):
    labels_combined = []
    for e in labels:
        labels_combined.extend(e)
    labels_unique = list(set(labels_combined))
    tag2idx = {v:i+1 for i,v in enumerate(labels_unique)}
    tag2idx['PAD'] = 0
    idx2tag = {i:v for v,i in tag2idx.items()}
    return tag2idx, idx2tag


def join(text, tags):
    new_text = []
    new_labels = []

    for sentence, tags_row in tqdm(zip(text, tags), total=len(text)):
        new_text_row = [sentence[0]]
        new_labels_row = [tags_row[0]]
        for i in range(1,len(sentence)):
            token = sentence[i]
            tag = tags_row[i]
            if tag!='O' and tag == tags_row[i-1]:
                if len(token) == 1 or token.startswith("'") or token[1]=="'":
                    if token!='a':
                        new_text_row[-1] = new_text_row[-1] + token
                else:
                    if sentence[i-1] == '-':
                        new_text_row[-1] = new_text_row[-1] + token
                    else:
                        new_text_row[-1] = new_text_row[-1] + ' ' + token
            else:
                new_text_row.append(token)
                new_labels_row.append(tag)

        new_text.append(new_text_row)
        new_labels.append(new_labels_row)

    return new_text, new_labels