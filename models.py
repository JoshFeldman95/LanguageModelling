import torch
from namedtensor import ntorch, NamedTensor
from collections import Counter
import numpy as np

class TrigramModel(object):
    def __init__(self, alpha1, alpha2, vocab_size, smoothing = 0.01):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.vocab_size = vocab_size
        self.smoothing = smoothing

    def fit(self, train_iter):
        self.ngram_counts = [self._get_ngram_counts(train_iter, n) for n in [1,2,3]]

    def predict(self, text, predict_last = False):
        """Make prediction on named tensor with dimensions 'batch' and 'seqlen'
        """
        batch = text.transpose('batch','seqlen').values.numpy()
        batch_size, text_len = batch.shape[0], batch.shape[1]
        predictions = np.zeros([batch_size, text_len, self.vocab_size])

        for batch_id, text in enumerate(batch):
            for word_id, word in enumerate(text):
                if predict_last and word_id != (len(text) - 1):
                    continue
                minus1 = word
                if (word_id-1) >= 0:
                    minus2 = text[word_id-1]
                else:
                    minus2 = None
                predictions[batch_id, word_id] = self._get_pred_dist(minus1, minus2)

        return NamedTensor(torch.from_numpy(predictions), names = ('batch','seqlen','distribution'))

    def _get_pred_dist(self, minus1, minus2):
        unigram_dist = np.zeros(self.vocab_size)
        bigram_dist = np.zeros(self.vocab_size)
        trigram_dist = np.zeros(self.vocab_size)

        for word in range(self.vocab_size):
            unigram_dist[word] = self.ngram_counts[0][(word,)]
            if minus1 is not None:
                bigram_dist[word] = self.ngram_counts[1][(minus1, word)]
                if minus2 is not None:
                    trigram_dist[word] = self.ngram_counts[2][(minus2, minus1, word)]

        #normalize
        unigram_dist = (unigram_dist+self.smoothing)/(unigram_dist.sum()+self.smoothing*self.vocab_size)
        bigram_dist = (bigram_dist+self.smoothing)/(bigram_dist.sum()+self.smoothing*self.vocab_size)
        trigram_dist = (trigram_dist+self.smoothing)/(trigram_dist.sum()+self.smoothing*self.vocab_size)
        return self.alpha1*trigram_dist + self.alpha2*bigram_dist + (1-self.alpha1-self.alpha2)*unigram_dist

    def _find_ngrams(self, input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def _get_ngram_counts(self, train_iter, n):
        #count ngrams
        count = Counter()
        for batch in iter(train_iter):
            for text in batch.text.transpose('batch','seqlen').values:
                text = text.tolist()
                ngrams = self._find_ngrams(text, n)
                count.update(ngrams)
        return count

class NeuralNetwork(ntorch.nn.Module):
    def __init__():
        pass

    def forward():
        pass

    def fit(self, train_iter):
        pass

    def predict(self, text, predict_last = False):
        pass

class LSTM(ntorch.nn.Module):
    def __init__(self, hidden_size, layers, TEXT, device = 'cpu'):
        super(LSTM, self).__init__()
        self.pretrained_embeddings = TEXT.vocab.vectors.to(device)
        self.embedding = torch.nn.Embedding.from_pretrained(self.pretrained_embeddings, freeze=True)
        self.lstm = torch.nn.LSTM(TEXT.vocab.vectors.shape[1], hidden_size, bidirectional=True)
        self.lstm_dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(2*hidden_size, len(TEXT.vocab.itos))

    def forward(self, x):
        x = self.embedding(x)
        x, (hn, cn) = self.lstm(x)
        x = self.lstm_dropout(x)
        x = self.linear(x)
        return x


    def fit(self, train_iter):
        pass

    def predict(self, text, predict_last = False):
        pass

class Extension(ntorch.nn.Module):
    def __init__():
        pass

    def forward():
        pass

    def fit():
        pass

    def predict(self, text, predict_last = False):
        pass
