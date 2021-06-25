import argparse
import os
from representation_network import RepresentationNetwork
from vocab_utils import load_vocab
from representation_evaluation import rename_layers
from classification_learning import initialize_token_embeddings
import torch
import random
import textwrap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from Bio import SeqIO


def load_references(reference_path):
    print(f'Loading references from {reference_path}')
    references = []
    ref_labels = []
    for ref_file in reference_path:
        reference_name = ref_file.split('/')[-1].split('.')[0]
        ref = SeqIO.parse(ref_file, 'fasta')
        ref = [(r.id, r.seq) for r in ref]
        references.append(ref)
        ref_labels.append(reference_name)
    return references, ref_labels


def sample(references, ref_names, samples_per_reference):
    samples = []
    names = []
    for ref, ref_name in zip(references, ref_names):
        i = 0
        weights = [len(r) for id, r in ref]
        while i < samples_per_reference:
            reference_genome = random.choices(ref, weights=weights, k=1)[0]
            reference = reference_genome[1]
            contig = reference_genome[0]
            print(len(reference), contig)
            start_position = random.randint(0, len(reference))
            chunk = textwrap.wrap(str(reference[start_position:start_position + 512*9]), 9)
            chunk_neigh = textwrap.wrap(str(reference[start_position + 5: start_position + 5 + 512*9]), 9)
            if len(chunk) == 0 or len(chunk[-1]) < 9 or len(chunk) == 0 or len(chunk_neigh[-1]) < 9:
                print(f"Skipping chunk")
                continue
            samples.extend([chunk, chunk_neigh])
            names.extend([contig]*2)
            i += 1
    return samples, names


def calculacte_similarity(representations, labels, samples_per_reference):
    """
    For each anchor sample, we calculate cosine similarity with its neighbour, average cosine similarity for all anchors
    from the same reference and average cosine similarity for anchors in each other reference
    :return:
    """
    i = 0
    similarities = {}
    while i < len(representations):
        ith_label = labels[i][:-1]
        anchor = np.array(representations[i], ndmin=2)
        neighbour = np.array(representations[i+1], ndmin=2)
        neigh_sim = cosine_similarity(anchor, neighbour)

        similar_anchors = []
        j = i + 2
        while j < len(representations) and ith_label in labels[j]:
            similar_anchors.append(representations[j])
            j += 2
        similar_anchors = np.array(similar_anchors)
        if len(similar_anchors) != 0:
            ref_sim = cosine_similarity(anchor, similar_anchors)
            ref_sim = np.mean(ref_sim)
        else:
            ref_sim = 0.0

        dist_sims = []
        if j < len(representations):
            for k in range(0, (len(representations) - j)//(2*samples_per_reference)):
                sim_k = np.array([representations[l] for l in range(j, j + 2*samples_per_reference, 2)])
                dist_sims.append(np.mean(cosine_similarity(anchor, sim_k)))
                j += 2*samples_per_reference
        else:
            dist_sims.append([0.0])

        similarities[labels[i]] = [neigh_sim, ref_sim, dist_sims]
        i += 2

    return similarities


def plot_representations(reps, labels, title, samples_per_ref, tgt):
    colors = ['skyblue', 'palevioletred', 'gold', 'mediumaquamarine', 'lightcoral', 'cornflowerblue', 'lightgreen', 'mediumpurple', 'darkorange', 'teal']
    markers = ['o', 'v', '^', 'p', '*', 'X', 'd', 'P', '<', '>']
    markers = markers[:samples_per_ref]
    print(reps)
    for j in range(7):
        for i in range(j*2*samples_per_ref, (j+1)*2*samples_per_ref, 2):
            x = reps[i:i+2, 0]
            y = reps[i:i+2, 1]
            print(x, y)
            plt.scatter(x, y, c=cnames[colors[j]], label=labels[i], marker=markers[(i//2) % samples_per_ref], alpha=0.65, s=40)
    plt.title(title)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='xx-small')
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

    references, ref_names = load_references(args.references)
    print(ref_names)
    sampled_chunks, reference_labels = sample(references, ref_names, args.samples_per_reference)
    eval_data = initialize_token_embeddings(vocab, sampled_chunks, args.seq_len)

    eval_representations = []
    for i in range(0, len(eval_data), 64):
        batch = torch.cat(eval_data[i:i + 64]).to(devices[0])
        original_lens = [512]*batch.size()[0]
        reps = embedding_model.forward(batch, original_lens).to('cpu').tolist()

        eval_representations.extend(reps)

    tsne_eval = TSNE(n_components=2).fit_transform(eval_representations)
    pca = PCA(n_components=2)
    pca.fit(eval_representations)

    pca_eval = pca.transform(eval_representations)
    print(f'PCA variance: {pca.explained_variance_ratio_}')

    similarities = calculacte_similarity(eval_representations, reference_labels, args.samples_per_reference)
    df = pd.DataFrame.from_dict(similarities, orient='index', columns=['Neighbour', 'Same reference', 'Different references'])
    df.to_csv(args.tgt_dir + 'finetuned_cosine_similarities.csv')

    plot_representations(tsne_eval, reference_labels, 't-SNE for validation data', args.samples_per_reference, args.tgt_dir + 'chunk_similarity_tsne_plot.png')
    plot_representations(pca_eval, reference_labels, 'PCA for validation data', args.samples_per_reference, args.tgt_dir + 'chunk_similarity_pca_plot.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--references', nargs='+')
    parser.add_argument('--vocab_path')
    parser.add_argument('--samples_per_reference', type=int, default=10)
    parser.add_argument('--tgt_dir')
    parser.add_argument('--fine_tuned', action='store_true')
    parser.add_argument('--gpu', type=str, nargs='+')
    parser.add_argument('--seq_len', type=int, default=512)
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