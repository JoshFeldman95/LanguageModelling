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

class _NeuralNetworkLM(ntorch.nn.Module):

    def __init__(self, TEXT, device):
        super().__init__()
        self.TEXT = TEXT
        self.device = device
        self.pretrained_emb = TEXT.vocab.vectors.to(device)
        self.embedding = ntorch.nn.Embedding.from_pretrained(self.pretrained_emb, freeze=True)

    def fit(self, train_iter, val_iter=[], lr = 1e-2, momentum = 0.9, batch_size = 128, epochs = 10, interval = 1):
        self.to(self.device)
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
                    outputs.transpose('batch','out','seqlen').values,
                    labels.transpose('batch','seqlen').values
                )
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % interval == interval-1:    # print every 2000 mini-batches
                    print('[epoch: {}, batch: {}] loss: {}'.format(epoch + 1, i + 1, running_loss / interval))
                    running_loss = 0.0

            running_loss = 0.
            val_count = 0.
            for i, data in enumerate(val_iter):
                inputs, labels = data.text, data.target
                outputs = self(inputs)
                loss = criterion(
                    outputs.transpose('batch','out','seqlen').values,
                    labels.transpose('batch','seqlen').values
                )
                running_loss += loss.item()
                val_count += 1
            print('Val loss: {}'.format(running_loss / val_count))

        print('Finished Training')

    def predict(self, text, predict_last = False):
        pred = self(text)
        return pred


class NeuralNetwork(_NeuralNetworkLM):
    def __init__(self, TEXT, kernel_size=3, hidden_size=50, dropout=.2, device='cuda'):
        super().__init__(TEXT, device)
        self.kernel = kernel_size
        self.conv = ntorch.nn.Conv1d(
            in_channels=self.pretrained_emb.shape[1],
            kernel_size=kernel_size,
            out_channels=hidden_size,
            padding = self.kernel - 1,
            stride=1
        ).spec("embedding", "seqlen", "conv")

        self.fc = ntorch.nn.Linear(hidden_size, len(TEXT.vocab.itos)).spec('conv', 'out')
        self.dropout = ntorch.nn.Dropout(dropout)

        #self.l2 = ntorch.nn.Linear(hidden_sizes[0], hidden_sizes[1]).spec('embedding')
        #self.l3 = ntorch.nn.Linear(hidden_sizes[1], len(TEXT.vocab.itos)).spec('embedding', 'out')

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x).relu()
        x = x[{"seqlen": slice(0, x.shape['seqlen'] - self.kernel + 1)}]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LSTM(_NeuralNetworkLM):
    def __init__(self, TEXT, hidden_size=50, layers=1, dropout = 0.2, device = 'cpu'):
        super().__init__(TEXT, device)
        self.lstm = ntorch.nn.LSTM(self.pretrained_emb.shape[1], hidden_size, bidirectional=True).spec("embedding", "seqlen", "lstm")
        self.lstm_dropout = ntorch.nn.Dropout(dropout)
        self.linear = ntorch.nn.Linear(2*hidden_size, len(TEXT.vocab.itos)).spec('lstm', 'out')

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = self.linear(x)
        return x


class Extension(ntorch.nn.Module):
    def __init__():
        pass

    def forward():
        pass

    def fit():
        pass

    def predict(self, text, predict_last = False):
        pass
