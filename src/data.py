import pandas as pd
import spacy


class Data:

    def __init__(self, entity_path, data_path):
        self.entity_path = entity_path
        self.data_path = data_path
        self.nlp = spacy.load('en_core_web_sm', disable=['lemmatizer', 'textcat', 'tagger', 'ner'])

    def get_entity_dictionary(self):
        df = pd.read_csv(self.entity_path, sep='\t')
        entities = df['entity'].tolist()
        labels = df['label'].tolist()
        entity_dict = {entity:label for entity, label in zip(entities, labels)}
        return entity_dict

    def preprocess_data(self):
        with open(self.data_path, 'r') as file:
            data = file.read()
            file.close()
        doc = self.nlp(data)
        tokens = []
        dependencies = []
        for sentence in doc.sents:
            sentence_tokens = []
            sentence_dependencies = []
            for token in sentence:
                sentence_tokens.append(token.text)
                sentence_dependencies.append(token.dep_)
            tokens.append(sentence_tokens)
            dependencies.append(sentence_dependencies)
        return tokens, dependencies        


