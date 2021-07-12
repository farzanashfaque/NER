import config
from utils import join_subtokens, join
import numpy as np
from tqdm import tqdm


def get_predictions(model, ids, mask, idx2tag):
    preds = model.predict([ids, mask], batch_size=config.BATCH_SIZE)
    preds = np.argmax(preds, axis=2) 
    ids = ids.tolist()
    preds = preds.tolist()

    for i in tqdm(range(len(ids))):
        idx = ids[i].index(102)
        ids[i] = ids[i][1:idx]
        preds[i] = preds[i][1:idx]

        for j in range(len(ids[i])):
            ids[i][j] = config.TOKENIZER.decode([ids[i][j]])
            preds[i][j] = idx2tag[preds[i][j]]
    
    joined_text, joined_labels = join_subtokens(ids, preds)
    joined_text, joined_labels = join(joined_text, joined_labels)

    return joined_text, joined_labels

    
    

