import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Dropout
from tensorflow.keras.metrics import Precision, Recall
from transformers import TFBertModel
import config


class EntityModel:
    def __init__(self):
        self.bert = TFBertModel.from_pretrained('bert-base-cased')
        self.num_tags = config.NUM_TAGS

    
    def model(self):
        max_len = config.MAX_LEN
        num_tags = self.num_tags
        input_ids = Input(shape=(max_len,), name='ids', dtype='int32')
        input_mask = Input(shape=(max_len,), name='mask', dtype='int32')
        bertOut = self.bert(input_ids=input_ids, attention_mask=input_mask)[0]
        out = Bidirectional(LSTM(256, return_sequences=True))(bertOut)
        out = TimeDistributed(Dense(num_tags, activation='softmax'))(out)
        model = Model(inputs=[input_ids, input_mask], outputs=out)
        for layer in model.layers[:3]:
            layer.trainable = False
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])
        return model