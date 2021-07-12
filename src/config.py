from transformers import BertTokenizer


MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 1
NUM_TAGS = 10
ITERATIONS = 5
TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)