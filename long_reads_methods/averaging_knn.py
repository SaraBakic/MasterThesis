import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from averaging_representation_network import RepresentationNetwork
from vocab_utils import load_vocab
from averaging_representation_evaluation import references_to_labels, get_labels, rename_layers, initialize_token_embeddings
from averaging_classification_learning import tokenize_chunks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


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

    train_df = pd.read_csv(args.train_chunks)
    if args.train_reads != None:
        train_reads = pd.read_csv(args.train_reads)
        print(f'Loaded {len(train_df)} chunks samples and {len(train_reads)} read samples')
        train_df = train_df.append(train_reads, ignore_index=True)
        train_df = train_df.sample(frac=1)
        train_df.reset_index(drop=True, inplace=True)
    train_data, train_lens = tokenize_chunks(list(train_df['Sequence']), 9)

    eval_df = pd.read_csv(args.eval_chunks)
    eval_data, eval_lens = tokenize_chunks(list(eval_df['Sequence']), 9)

    seq_to_label, label_to_seq = references_to_labels(list(train_df['General_ref']))
    eval_labels = get_labels(list(eval_df['General_ref']), seq_to_label)
    train_labels = get_labels(list(train_df['General_ref']), seq_to_label)
    print(seq_to_label)

    train_representations = []
    for i in tqdm(range(0, len(train_data), args.batch_size)):
        train_batch, train_slices = initialize_token_embeddings(vocab, train_data[i:i+args.batch_size], args.seq_len)
        batch = torch.cat(train_batch).to(devices[0])
        lens = np.concatenate(train_lens[i:i + args.batch_size])
        #print(batch.size(), lens, train_slices)
        reps = embedding_model.forward(batch, lens, train_slices).to('cpu').tolist()

        train_representations.extend(reps)

    eval_representations = []
    for i in tqdm(range(0, len(eval_data), args.batch_size)):
        eval_batch, eval_slices = initialize_token_embeddings(vocab, eval_data[i:i+args.batch_size], args.seq_len)
        batch = torch.cat(eval_batch).to(devices[0])
        lens = np.concatenate(eval_lens[i:i + args.batch_size])
        #print(batch.size(), lens, eval_slices)
        reps = embedding_model.forward(batch, lens, eval_slices).to('cpu').tolist()

        eval_representations.extend(reps)

    for neigh in range(2, 16):
        neigh_classifier = KNeighborsClassifier(n_neighbors=neigh)
        neigh_classifier.fit(train_representations, train_labels)

        eval_predictions = neigh_classifier.predict(eval_representations)

        conf_matrix = confusion_matrix(eval_labels, eval_predictions)
        accuracy = accuracy_score(eval_labels, eval_predictions)
        recall = recall_score(eval_labels, eval_predictions, average='macro')
        precision = precision_score(eval_labels, eval_predictions, average='macro')
        f1 = f1_score(eval_labels, eval_predictions, average='macro')

        print(f'Evaluation results with k={neigh}: accuracy = {accuracy}, macro recall = {recall}, macro precision = {precision}, macro F1 = {f1}, confusion matrix = \n')
        print(f'{np.array2string(conf_matrix, separator=",")}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_chunks')
    parser.add_argument('--train_reads')
    parser.add_argument('--eval_chunks')
    parser.add_argument('--model_path')
    parser.add_argument('--vocab_path')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seq_len', type=int, default=1024)
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
