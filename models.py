import torch
from namedtensor import ntorch, NamedTensor
from collections import Counter
import numpy as np
import math


class TrigramModel(object):
    def __init__(self, alpha1, alpha2, vocab_size, smoothing=0.01):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.vocab_size = vocab_size
        self.smoothing = smoothing

    def fit(self, train_iter):
        self.ngram_counts = [self._get_ngram_counts(train_iter, n) for n in [1, 2, 3]]

    def predict(self, text, predict_last=False):
        """Make prediction on named tensor with dimensions 'batch' and 'seqlen'
        """
        batch = text.transpose("batch", "seqlen").values.numpy()
        batch_size, text_len = batch.shape[0], batch.shape[1]
        predictions = np.zeros([batch_size, text_len, self.vocab_size])

        for batch_id, text in enumerate(batch):
            for word_id, word in enumerate(text):
                if predict_last and word_id != (len(text) - 1):
                    continue
                minus1 = word
                if (word_id - 1) >= 0:
                    minus2 = text[word_id - 1]
                else:
                    minus2 = None
                predictions[batch_id, word_id] = self._get_pred_dist(minus1, minus2)

        return NamedTensor(
            torch.from_numpy(predictions), names=("batch", "seqlen", "distribution")
        )

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

        # normalize
        unigram_dist = (unigram_dist + self.smoothing) / (
            unigram_dist.sum() + self.smoothing * self.vocab_size
        )
        bigram_dist = (bigram_dist + self.smoothing) / (
            bigram_dist.sum() + self.smoothing * self.vocab_size
        )
        trigram_dist = (trigram_dist + self.smoothing) / (
            trigram_dist.sum() + self.smoothing * self.vocab_size
        )
        return (
            self.alpha1 * trigram_dist
            + self.alpha2 * bigram_dist
            + (1 - self.alpha1 - self.alpha2) * unigram_dist
        )

    def _find_ngrams(self, input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def _get_ngram_counts(self, train_iter, n):
        # count ngrams
        count = Counter()
        for batch in iter(train_iter):
            for text in batch.text.transpose("batch", "seqlen").values:
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
        self.embedding = ntorch.nn.Embedding.from_pretrained(
            self.pretrained_emb, freeze=True
        )

    def fit(
        self,
        train_iter,
        val_iter=[],
        lr=1e-2,
        momentum=0.9,
        batch_size=128,
        epochs=10,
        interval=1,
    ):
        self.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        train_iter.batch_size = batch_size

        for epoch in range(epochs):  # loop over the dataset multiple times
            self.train()
            running_loss = 0.0
            for i, data in enumerate(train_iter, 0):
                # get the inputs
                inputs, labels = data.text, data.target

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(
                    outputs.transpose("batch", "out", "seqlen").values,
                    labels.transpose("batch", "seqlen").values,
                )
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % interval == interval - 1:  # print every 2000 mini-batches
                    print(
                        "[epoch: {}, batch: {}] loss: {}".format(
                            epoch + 1, i + 1, running_loss / interval
                        )
                    )
                    running_loss = 0.0

            running_loss = 0.0
            val_count = 0.0
            self.eval()
            for i, data in enumerate(val_iter):
                inputs, labels = data.text, data.target
                outputs = self(inputs)
                loss = criterion(
                    outputs.transpose("batch", "out", "seqlen").values,
                    labels.transpose("batch", "seqlen").values,
                )
                running_loss += loss.item()
                val_count += 1
            print("Val loss: {}".format(running_loss / val_count))

        print("Finished Training")

    def predict(self, text, predict_last=False):
        pred = self(text)
        return pred


class NeuralNetwork(_NeuralNetworkLM):
    def __init__(self, TEXT, kernel_size=3, hidden_size=50, dropout=0.2, device="cuda"):
        super().__init__(TEXT, device)
        self.kernel = kernel_size
        self.conv = ntorch.nn.Conv1d(
            in_channels=self.pretrained_emb.shape[1],
            kernel_size=kernel_size,
            out_channels=hidden_size,
            padding=self.kernel - 1,
            stride=1,
        ).spec("embedding", "seqlen", "conv")

        self.fc = ntorch.nn.Linear(hidden_size, len(TEXT.vocab.itos)).spec(
            "conv", "out"
        )
        self.dropout = ntorch.nn.Dropout(dropout)

        # self.l2 = ntorch.nn.Linear(hidden_sizes[0], hidden_sizes[1]).spec('embedding')
        # self.l3 = ntorch.nn.Linear(hidden_sizes[1], len(TEXT.vocab.itos)).spec('embedding', 'out')

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x).relu()
        x = x[{"seqlen": slice(0, x.shape["seqlen"] - self.kernel + 1)}]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LSTM(_NeuralNetworkLM):
    def __init__(self, TEXT, hidden_size=50, layers=1, dropout=0.2, device="cpu"):
        super().__init__(TEXT, device)
        self.lstm = ntorch.nn.LSTM(
            self.pretrained_emb.shape[1], hidden_size, bidirectional=True
        ).spec("embedding", "seqlen", "lstm")
        self.lstm_dropout = ntorch.nn.Dropout(dropout)
        self.linear = ntorch.nn.Linear(2 * hidden_size, len(TEXT.vocab.itos)).spec(
            "lstm", "out"
        )

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = self.linear(x)
        return x


class MultiHeadAttention(ntorch.nn.Module):
    def __init__(
        self,
        input_depth,
        total_key_depth,
        total_value_depth,
        output_depth,
        num_heads,
        bias_mask,
        dropout=0,
    ):
        super(MultiHeadAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            raise ValueError(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_depth, num_heads)
            )
        if total_value_depth % num_heads != 0:
            raise ValueError(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_depth, num_heads)
            )

        self.query_linear = ntorch.nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = ntorch.nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = ntorch.nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = ntorch.nn.Linear(
            total_value_depth, output_depth, bias=False
        )

        self.dropout = ntorch.nn.Dropout(dropout)
        self.num_heads = num_heads

        self.query_scale = (total_key_depth // num_heads) ** -0.5

        self.bias_mask = bias_mask

    def forward(self, queries, keys, values):
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # rename queries and keys
        queries = queries.rename("seqlen", "queries")

        # Scale queries
        queries = queries * self.query_scale

        # Combine queries and keys
        logits = queries.dot("embedding", keys)

        # Add bias to mask future values
        if self.bias_mask is not None:
            mask_slice = {
                "queries": slice(0, logits.shape["queries"]),
                "seqlen": slice(0, logits.shape["seqlen"]),
            }
            logits += self.bias_mask[mask_slice]

        # Convert to probabilites
        weights = logits.softmax("seqlen")
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = weights.dot("seqlen", values)

        # Merge heads
        contexts = self._merge_heads(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        return outputs

    def _split_heads(self, x):
        return x.split("embedding", ("heads", "embedding"), heads=self.num_heads)

    def _merge_heads(self, x):
        return x.stack(("heads", "embedding"), "embedding")


class PositionwiseFeedForward(ntorch.nn.Module):
    def __init__(self, d_model, filter_size, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.c1 = ntorch.nn.Conv1d(d_model, filter_size, 1).spec(
            "embedding", "seqlen", "hidden"
        )
        self.c2 = ntorch.nn.Conv1d(filter_size, d_model, 1).spec(
            "hidden", "seqlen", "embedding"
        )

    def forward(self, x):
        x = self.c1(x).relu()
        return self.c2(x)


class LayerNorm(ntorch.nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        # ntorch.nn.Parameter(ntorch.ones(features))
        # ntorch.nn.Parameter(ntorch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean("embedding")
        std = x.std("embedding")
        # return self.gamma * (x - mean) / (std + self.eps) + self.beta
        return (x - mean) / (std + self.eps)


class EncoderLayer(ntorch.nn.Module):
    def __init__(
        self,
        hidden_size,
        total_key_depth,
        total_value_depth,
        filter_size,
        num_heads,
        bias_mask=None,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        """
    Parameters:
        hidden_size: Hidden size
        total_key_depth: Size of last dimension of keys. Must be divisible by num_head
        total_value_depth: Size of last dimension of values. Must be divisible by num_head
        output_depth: Size last dimension of the final output
        filter_size: Hidden size of the middle layer in FFN
        num_heads: Number of attention heads
        bias_mask: Masking tensor to prevent connections to future elements
        layer_dropout: Dropout for this layer
        attention_dropout: Dropout probability after attention (Should be non-zero only during training)
        relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
    """

        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            bias_mask,
            attention_dropout,
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size, filter_size, dropout=relu_dropout
        )
        self.dropout = ntorch.nn.Dropout(layer_dropout)
        self.layer_norm_attn = LayerNorm(hidden_size)
        self.layer_norm_feedforward = LayerNorm(hidden_size)

    def forward(self, x):

        # Multi-head attention
        y = self.attn(x, x, x)
        # Dropout and residual
        x = self.dropout(x + y.rename("queries", "seqlen"))

        # Layer Normalization
        x_norm = self.layer_norm_attn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        # Dropout and residual
        y = self.dropout(x + y)
        # Layer Normalization
        y_norm = self.layer_norm_feedforward(y)

        return y


class Transformer(_NeuralNetworkLM):
    def __init__(
        self,
        TEXT,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=32,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        device="cuda",
    ):
        """
        Parameters:
            TEXT: torchtext object
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        super().__init__(TEXT, device)

        hidden_size = self.pretrained_emb.shape[1]

        self.timing_signal = self._gen_timing_signal(max_length, hidden_size)


        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            self._gen_bias_mask(max_length).to(device) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.enc = [EncoderLayer(*params).to(device) for l in range(num_layers)]

        self.layer_norm = LayerNorm(hidden_size)

        self.word_proj = ntorch.nn.Linear(hidden_size, len(self.TEXT.vocab)).spec(
            "embedding", "out"
        )

    def forward(self, x):
        x = self.embedding(x)

        # Add timing signal
        x = x + self.timing_signal[{"seqlen": slice(0, x.shape["seqlen"])}]

        # Encoder blocks
        for encoder in self.enc:
            x = encoder(x)

        # Final Layer Norm
        x = self.layer_norm(x)

        # Get next word probability logits
        x = self.word_proj(x)

        return x

    def _gen_bias_mask(self, max_length):
        """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
        np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
        torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
        torch_mask = NamedTensor(torch_mask, names=("queries", "seqlen"))
        return torch_mask

    def _gen_timing_signal(
        self,
        length,
        channels,
        min_timescale=1.0,
        max_timescale=1e4
    ):
        """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
        position = np.arange(length)
        num_timescales = channels // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (float(num_timescales) - 1)
        inv_timescales = min_timescale * np.exp(
            np.arange(num_timescales).astype(np.float) * -log_timescale_increment
        )
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.pad(
            signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0]
        )
        signal = signal.reshape([length, channels])

        return NamedTensor(
            torch.from_numpy(signal).type(torch.FloatTensor).to(self.device),
            names=("seqlen", "embedding"),
        )
