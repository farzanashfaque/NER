from utils import encode
from tensorflow import keras
import argparse
import spacy
import pickle
import config
from predict import get_predictions
import numpy as np


parser = argparse.ArgumentParser(description='Provide paths to saved model and document')
parser.add_argument('-m', '--model_path', type='str', metavar='', required=True, help='Model path relative to the inference script')
parser.add_argument('-d', '--data_path', type='str', metavar='', required=True, help='Data path relative to the inference script')
args = parser.parse_args()


def preprocess_data(data_path):
    with open(data_path, 'r') as file:
        data = file.read()
        file.close()
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'lemmatizer', 'textcat'])
    doc = nlp(data)
    tokens = []
    for sentence in doc.sents:
        sentence_tokens = []
        sentence_dependencies = []
        for token in sentence:
            sentence_tokens.append(token.text)
            sentence_dependencies.append(token.dep_)
        tokens.append(sentence_tokens)
    return tokens


def encode_data(data):
    with open('../label-encodings/tag2idx.pickle', 'rb') as handle:
        tag2idx = pickle.load(handle)
    max_len = config.MAX_LEN
    data_encoded = []
    tokenizer = config.TOKENIZER
    for sentence_tokens in tokens:
        encoded_sentence = []
        for token in sentence_tokens:
            token_encoded = tokenizer.encode(token)
            encoded_sentence.extend(token_encoded)
        encoded_sentence = encoded_sentence[:max_len-2]
        encoded_sentence = [101] + encoded_sentence + [102]
        num = max_len - len(encoded_sentence)
        encoded_sentence += [0]*num
        data_encoded.append(encoded_sentence)
    mask = [[int(i!=0) for i in ii] for ii in data_encoded]
    data_encoded = np.array(data_encoded)
    mask = np.array(mask)
    return data_encoded, mask


if __name__ == '__main__':
    model_path = args.model_path
    model = keras.models.load_model(model_path)
    data_path = args.data_path
    tokens = preprocess_data(data_path)
    data_encoded, mask = encode_data(tokens)
    with open('../label-encoding/idx2tag.pickle', 'rb') as handle:
        idx2tag = pickle.load(handle)
    text, tags = get_predictions(model, data_encoded, mask, idx2tag)
    pred_dict = {}
    for entity, tag in zip(text, tags):
        if tag not in pred_dict:
            pred_dict[tag] = [entity]
        else:
            pred_dict[tag].append(entity)
    for entity_type, entities in pred_dict:
        with open(f'../predictions/{entity_type}.txt', 'w') as file:
            for entity in entities:
                file.write(entity+'\n')
            file.close()
    