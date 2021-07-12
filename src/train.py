from labeling import Label
from model import EntityModel
import config
from dataset import preprocess_data
from data import Data
from utils import prepare_data, expand_entity_dict, label_encoding
from predict import get_predictions
import argparse
import pickle


# parser = argparse.ArgumentParser(description='Provide paths to entity.tsv and data file')
# parser.add_argument('-e', '--entity_path', type=str, metavar='', required=True, help='Path of entities tsv file')
# parser.add_argument('-d', '--data_path', type=str, metavar='', required=True, help='Path of data text file')
# args = parser.parse_args()


if __name__ == "__main__":
    # data = Data(args.entity_path, args.data_path)
    # entity_dict = data.get_entity_dictionary()
    # text, dep = data.preprocess_data()
    text, tags, entity_dict = preprocess_data()
    tag2idx, idx2tag = label_encoding(tags)
    with open('../label-encodings/tag2idx.pickle', 'wb') as handle:
        pickle.dump(tag2idx, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../label-encodings/idx2tag.pickle', 'wb') as handle:
        pickle.dump(idx2tag, protocol=pickle.HIGHEST_PROTOCOL)
    labeler = Label(text, entity_dict)
    dep = labeler.dependency_parse()
    labels = labeler.get_labels()
    labels = labeler.relabel(labels, dep)
    text_, mask, tags_ = prepare_data(text, labels, tag2idx)
    model = EntityModel().model()

    for i in range(config.ITERATIONS):
        model.fit([text_, mask], tags_, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS)
        model.save(f'../model/iter_{i}.h5')
        text_pred, labels_pred = get_predictions(model, text_, mask, idx2tag)
        if i == config.ITERATIONS - 1:
            break
        entity_dict = expand_entity_dict(entity_dict, text_pred, labels_pred, threshold=5)
        labeler = Label(text, entity_dict)
        labels = labeler.get_labels()
        labels = labeler.relabel(labels, dep)
        text_, mask, tags_ = prepare_data(text, labels)

    

    
        
    

