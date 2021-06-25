from averaging_representation_network import RepresentationNetwork
from vocab_utils import load_vocab
from averaging_classification_learning import tokenize_chunks
import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from mpl_toolkits import mplot3d
import numpy as np
from collections import OrderedDict
import math


def references_to_labels(sequences):
    seq_to_label = {}
    label_to_seq = {}
    for i, seq in enumerate(set(sequences)):
        seq_to_label[seq] = i
        label_to_seq[i] = seq
    return seq_to_label, label_to_seq


def get_labels(sequences, ref_to_label):
    labels = list(map(lambda seq: ref_to_label[seq], sequences))
    return torch.tensor(labels, dtype=torch.long)


def initialize_token_embeddings(vocab, tokenized_sequences, seq_len):
    """

    :return: Returns a list of long tensors where each tensor represents a chunk and each value in the tensor is vocab
    index of a "chunk token"
    """
    #data = [torch.unsqueeze(torch.tensor([vocab[kmer] if (kmer in vocab.keys()) else vocab['<unk>'] for kmer in seq], dtype=torch.long), 0) for seq in tokenized_sequences]
    data = []
    slices = []
    for seq in tokenized_sequences:
        t = torch.unsqueeze(torch.tensor([vocab[kmer] if (kmer in vocab.keys()) else vocab['<unk>'] for kmer in seq], dtype=torch.long), 0)
        if t.size()[1] > seq_len:
            splitted_t = list(torch.split(t, seq_len, 1))
            if splitted_t[-1].size()[1] < 0.65*seq_len:
                splitted_t = splitted_t[:-1]
            elif splitted_t[-1].size()[1] < seq_len:
                padding = torch.tensor([[vocab['<pad>']] * (seq_len - splitted_t[-1].size()[1])], dtype=torch.long)
                splitted_t[-1] = torch.cat((splitted_t[-1], padding), dim=1)
            splitted_t = torch.cat(splitted_t, dim=0)
            data.append(splitted_t)
            slices.append(splitted_t.size()[0])
        elif t.size()[1] < seq_len:
            padding = torch.tensor([[vocab['<pad>']]*(seq_len - t.size()[1])], dtype=torch.long)
            data.append(torch.cat((t, padding), dim=1))
            slices.append(1)
        else:
            data.append(t)
            slices.append(1)
    #return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    return data, slices


def rename_layers(model):
    new_state_dict = OrderedDict()
    for key, val in model.items():
        if "embedding_module." in key:
            new_key = key.replace('embedding_module.', '')
            new_state_dict[new_key] = val

    return new_state_dict


def select_data(projections, labels, tgt_label):
    selected = [p for (p, l) in zip(projections, labels) if l == tgt_label]
    return np.array(selected)


def plot_projected_data(projections, labels, title, tgt, labels_to_seq):
    colors = ['skyblue', 'palevioletred', 'gold', 'mediumaquamarine', 'lightcoral', 'cornflowerblue', 'lightgreen', 'mediumpurple', 'darkorange', 'teal']

    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    for label, seq in labels_to_seq.items():
        data = select_data(projections, labels, label)
        x = data[:, 0]
        y = data[:, 1]
        #z = data[:, 2]
        plt.scatter(x, y, c=cnames[colors[label]], label=seq, alpha=0.65)
    #ax.view_init(15, 0)
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='xx-small')
    plt.savefig(tgt, bbox_inches='tight')
    plt.close()


def main(args, devices):
    embedding_model_checkpoint = torch.load(args.model_path)
    vocab = load_vocab(args.vocab_path)
    embedding_model = RepresentationNetwork(len(vocab), 192, 8, 192, 0.1, 6, devices[0])
    if args.fine_tuned:
        embedding_model_checkpoint = rename_layers(embedding_model_checkpoint.state_dict())
        embedding_model.load_state_dict(embedding_model_checkpoint)
    else:
        embedding_model.load_state_dict(embedding_model_checkpoint.state_dict())
    embedding_model.to(devices[0])
    embedding_model.eval()

    print(embedding_model.state_dict)

    eval_df = pd.read_csv(args.chunks)
    eval_data, eval_lens = tokenize_chunks(list(eval_df['Sequence']), 9)
    #original_lens = [len(d) if len(d) < 1024 else 1024 for d in eval_data]
    full_lens = list(eval_df['Length'])

    seq_to_label, label_to_seq = references_to_labels(list(eval_df['General_ref']))
    print(seq_to_label)

    eval_labels = get_labels(list(eval_df['General_ref']), seq_to_label)

    eval_representations = []
    for i in range(0, len(eval_data), args.batch_size):
        eval_batch, eval_slices = initialize_token_embeddings(vocab, eval_data[i:i+args.batch_size], args.seq_len)
        eval_batch = torch.cat(eval_batch)
        eval_batch = eval_batch.to(devices[0])
        lens = np.concatenate(eval_lens[i:i+args.batch_size])
        #print(eval_batch.size(), eval_slices, lens)
        reps = embedding_model.forward(eval_batch, lens, eval_slices).to('cpu').tolist()
        eval_representations.extend(reps)

    tsne_eval = TSNE(n_components=2).fit_transform(eval_representations)
    pca = PCA(n_components=2)
    pca.fit(eval_representations)

    pca_eval = pca.transform(eval_representations)
    print(f'PCA variance: {pca.explained_variance_ratio_}')

    plot_projected_data(tsne_eval, eval_labels, 't-SNE for validation data',
                        args.plots_tgt + 'avg_class_representation_tsne.png', label_to_seq)
    plot_projected_data(pca_eval, eval_labels, 'PCA for validation data',
                        args.plots_tgt + 'avg_class_representation_pca.png', label_to_seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks')
    parser.add_argument('--model_path')
    parser.add_argument('--vocab_path')
    parser.add_argument('--plots_tgt')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--fine_tuned', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
        devices = [torch.device('cuda', i) for i in range(len(args.gpu))]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        devices = [torch.device('cpu')]

    print(f'Device set to {devices[0]}')

    main(args, devices)