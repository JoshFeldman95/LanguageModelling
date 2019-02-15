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
    def __init__(self, TEXT, hidden_size=50, layers=1, dropout = 0.2, device = 'cpu'):
        super(LSTM, self).__init__()
        self.TEXT = TEXT
        self.pretrained_emb = TEXT.vocab.vectors.to(device)
        self.embedding = ntorch.nn.Embedding.from_pretrained(self.pretrained_emb, freeze=True)
        self.lstm = ntorch.nn.LSTM(self.pretrained_emb.shape[1], hidden_size, bidirectional=True).spec("embedding", "seqlen", "lstm")
        self.lstm_dropout = ntorch.nn.Dropout(dropout)
        self.linear = ntorch.nn.Linear(2*hidden_size, len(TEXT.vocab.itos)).spec('lstm', 'out')

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = self.linear(x)
        return x

    def fit(self, train_iter, lr = 1e-2, momentum = 0.9, batch_size = 128, epochs = 10, interval = 1, device = 'cpu'):
        self.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        train_iter.batch_size = batch_size

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_iter, 0):
                # get the inputs
                inputs, labels = data.text, data.target

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(
                    outputs.transpose('batch', 'out', 'seqlen').values,
                    labels.transpose('batch','seqlen').values
                )
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % interval == interval-1:    # print every 2000 mini-batches
                    print('[epoch: {}, batch: {}] loss: {}'.format(epoch + 1, i + 1, running_loss / interval))
                running_loss = 0.0

        print('Finished Training')

    def predict(self, text, predict_last = False):
        pred = self(text)
        return pred

class Extension(ntorch.nn.Module):
    def __init__():
        pass

    def forward():
        pass

    def fit():
        pass

    def predict(self, text, predict_last = False):
        pass
