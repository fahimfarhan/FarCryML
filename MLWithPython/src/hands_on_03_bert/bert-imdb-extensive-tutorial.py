import sys
import numpy as np
import random as rn
import pandas as pd
import torch
from keras.src.utils import pad_sequences
from pytorch_pretrained_bert import BertModel
from tensorflow.python.ops.control_flow_util_v2 import output_all_intermediates
from torch import nn
# from torchnlp.datasets import imdb_dataset      # --> We are using our own uploaded dataset.
from pytorch_pretrained_bert import BertTokenizer
# from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
# from IPython.display import clear_output
import matplotlib.pyplot as plt

rn.seed(321)
np.random.seed(321)
torch.manual_seed(321)
torch.cuda.manual_seed(321)

path = "./data/"

train_data = pd.read_csv(f"{path}train.csv")
test_data = pd.read_csv(f"{path}test.csv")

# train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), train_data)))
# test_texts, test_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), test_data)))

train_texts = train_data["text"]
train_labels = train_data["sentiment"]

test_texts = test_data["text"]
test_labels = test_data["sentiment"]

print(f"{len(train_texts), len(train_labels), len(test_texts), len(test_labels)}")

print(f"{train_texts[0] = }")
print(f"{train_labels[0] = }")

# visualize the graph
"""
ySentenceLengths = [len(sent) for sent in train_texts]
xSentenceIndices = [i for i in range(0, len(train_texts) )]
# x = sentence idx,
# y = sentence length

# plot graph of sentence length to sentence idx, ie, ith sentence has length ith len
plt.bar(xSentenceIndices, ySentenceLengths)
plt.title("graph of sentence length to sentence idx")
plt.xlabel("sentence idx")
plt.ylabel("sentence length")

plt.show()
x = input("Type anything to continue...")
print("exit...")
"""

MODEL_NAME = "bert-base-uncased" # "distilbert-base-uncased" <-- not supported by pytorch_pretrained_bert library

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME, do_lower_case=True)

sample_tokenized = tokenizer.tokenize("Hi My name is Soumic")
print(sample_tokenized)

train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], train_texts))
test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:510] + ['[SEP]'], test_texts))

print(f"{len(train_tokens) = }, {len(test_tokens) = }")



train_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, train_tokens)), maxlen=512, truncating="post", padding="post", dtype="int")
test_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, test_tokens)), maxlen=512, truncating="post", padding="post", dtype="int")

print(f"{train_tokens_ids.shape = }, {test_tokens_ids.shape = }")

train_y = np.array(train_labels) == 'pos'
test_y = np.array(test_labels) == 'pos'
train_y.shape, test_y.shape, np.mean(train_y), np.mean(test_y)

# Now Masking few random IDs from each sentences to remove Biasness from model.¶
train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

# BaseLine:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report



baseline_model = make_pipeline(CountVectorizer(ngram_range=(1,3)), LogisticRegression()).fit(train_texts, train_labels)
baseline_predicted = baseline_model.predict(test_texts)

print(classification_report(test_labels, baseline_predicted))


class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout = 0.01):
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, tokens, masks=None):
        ignore, pooled_output = self.bert(tokens, attention_masks=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        logits_linear_output = self.linear(dropout_output)
        probability = self.sigmoid(logits_linear_output)
        return probability

    # ensuring that the model runs on GPU, not on CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

bert_clf = BertBinaryClassifier()
bert_clf = bert_clf.cuda()     # running BERT on CUDA_GPU
cudaMemoryAllocated = str(torch.cuda.memory_allocated(device)/1000000 ) + 'M'
print(f"{cudaMemoryAllocated = }")



x = torch.tensor(train_tokens_ids[:3]).to(device)
y, pooled = bert_clf.bert(x, output_all_encoded_layers=False)
print(f"{x.shape = }, {y.shape = }, {pooled.shape = }")

probability = bert_clf.forward(tokens=x)
print(f"{probability = }")
# Fine Tune BERT too foreign. this guy manually trained his model, maybe huggingface transformer wasn't so popular back then...
# let's stick to one, and only one thing.
