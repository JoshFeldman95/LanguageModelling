"""functions to train models. Running main will train models and save performance
stats.
"""
import torch
import numpy as np

from namedtensor import ntorch, NamedTensor

def make_kaggle_submission(model, TEXT, path_to_data = "./data/", device = 'cpu'):
    kaggle_input = load_kaggle_data(path_to_data+"/input.txt", TEXT, device)
    pred = model.predict(kaggle_input, predict_last=True)

    _, top20 = pred[{'seqlen':-1}].values.topk(20, dim = 1)

    with open(path_to_data+"/sample.txt", "w") as fout:
        print("id,word", file=fout)
        for i, text in enumerate(top20, 1):
            predictions = [TEXT.vocab.itos[word] for word in text]
            print("%d,%s"%(i, " ".join(predictions)), file=fout)

def load_kaggle_data(path_to_data, TEXT, device):
    with open(path_to_data) as f:
        data = f.read()
    sentences = [sent for sent in data.split('\n')[:-1]]
    convert_sent_to_int = lambda sent: [TEXT.vocab.stoi[word] for word in sent.split(' ')[:-1]]
    sent_list = np.array([convert_sent_to_int(sent) for sent in sentences])
    return NamedTensor(torch.from_numpy(sent_list).to(device), names = ('batch','seqlen'))
