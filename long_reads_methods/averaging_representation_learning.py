import argparse
import os
import torch
import math
import random
import pandas as pd
from tqdm import tqdm
from collections import Counter
from torchtext.vocab import Vocab
from initial_kmer_embeddings import tokenize_chunks
from averaging_representation_network import RepresentationNetwork
from vocab_utils import load_vocab


def initialize_token_embeddings(vocab, tokenized_sequences, seq_len):
    """

    :return: Returns a list of long tensors where each tensor represents a chunk and each value in the tensor is vocab
    index of a "chunk token"
    """
    #data = [torch.unsqueeze(torch.tensor([vocab[kmer] if (kmer in vocab.keys()) else vocab['<unk>'] for kmer in seq], dtype=torch.long), 0) for seq in tokenized_sequences]
    data = []
    slices = []
    for seq in tokenized_sequences:
        t = torch.unsqueeze(torch.tensor([vocab[kmer] if (kmer in vocab.stoi.keys()) else vocab['<unk>'] for kmer in seq], dtype=torch.long), 0)
        if t.size()[1] > seq_len:
            splitted_t = list(torch.split(t, seq_len, 1))
            if splitted_t[-1].size()[1] < seq_len:
                splitted_t = splitted_t[:-1]
                """padding = torch.tensor([[vocab['<pad>']] * (seq_len - splitted_t[-1].size()[1])], dtype=torch.long)
                splitted_t[-1] = torch.cat((splitted_t[-1], padding), dim=1)"""
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
    for i in range(10):
        print(data[i].size())
    #return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    return data, slices


def create_vocab(tokenized_chunks):
    counter = Counter()
    for chunk in tokenized_chunks:
        counter.update(chunk)
    return Vocab(counter)


def save_vocab(vocab, path):
    with open(path, 'w') as f:
        for token, index in vocab.stoi.items():
            f.write(f'{token}\t{index}\n')


def get_positives_negatives(anchor_indices, train_dfs):
    """
    For a given anchor we find a positive and a negative sample where positive samples are those chunks sampled starting
    from a position within +/- N bases from the anchor starting position in the same reference
    :param anchor_indices:
    :param train_dfs:
    :return:
    """
    anchors = []
    positives = []
    negatives = []
    anchor_lens = []
    positives_lens = []
    negatives_lens = []
    anchor_slices = []
    positive_slices = []
    negative_slices = []
    for anchor in anchor_indices:
        tgt = train_dfs[train_dfs['Index'] == anchor]
        if len(tgt.index) != 1:
            continue
        reference = list(tgt['Reference'])[0]
        general_reference = list(tgt['General_ref'])[0]
        start_position = list(tgt['Start position'])[0]
        index = list(tgt['Index'])[0]
        anchor_slice = list(tgt['Slice_lengths'])[0]
        potential_positives = train_dfs.loc[(train_dfs['Reference'] == reference) & ((train_dfs['Start position'] - start_position).abs() < 150) & (train_dfs['Index'] != index)]

        potential_negatives = train_dfs.loc[(train_dfs['General_ref'] != general_reference) & ((train_dfs['Start position'] - start_position).abs() < 150)]
        if len(potential_positives) == 0 or len(potential_negatives) == 0:
            continue
        pos = potential_positives.sample(n=1)
        neg = potential_negatives.sample(n=1)
        positives.append(list(pos['Tokenized_sequences'])[0])
        lens = [512] * (list(pos['Length'])[0] // (512 * 9)) + [int(math.ceil(list(pos['Length'])[0] % (512 * 9) / 9))]
        if lens[-1] == 0:
            lens = lens[:-1]
        if len(lens) > 1 and lens[-1] != 512:
            lens = lens[:-1]
        positives_lens.extend(lens)
        positive_slices.append(list(pos['Slice_lengths'])[0])
        negatives.append(list(neg['Tokenized_sequences'])[0])
        lens = [512] * (list(neg['Length'])[0] // (512 * 9)) + [int(math.ceil(list(neg['Length'])[0] % (512 * 9) / 9))]
        if lens[-1] == 0:
            lens = lens[:-1]
        if len(lens) > 1 and lens[-1] != 512:
            lens = lens[:-1]
        negatives_lens.extend(lens)
        negative_slices.append(list(neg['Slice_lengths'])[0])
        anchors.append(list(tgt['Tokenized_sequences'])[0])
        lens = [512] * (list(tgt['Length'])[0] // (512 * 9)) + [int(math.ceil(list(tgt['Length'])[0] % (512 * 9) / 9))]
        if lens[-1] == 0:
            lens = lens[:-1]
        if len(lens) > 1 and lens[-1] != 512:
            lens = lens[:-1]
        anchor_lens.extend(lens)
        anchor_slices.append(anchor_slice)


    return torch.cat(anchors), torch.cat(positives), torch.cat(negatives), anchor_lens, positives_lens, negatives_lens, \
           anchor_slices, positive_slices, negative_slices


def main(args, devices):
    train_df = pd.read_csv(args.train_chunks)
    if args.train_reads != None:
        train_reads = pd.read_csv(args.train_reads)
        print(f'Loaded {len(train_df)} chunks samples and {len(train_reads)} read samples')
        train_df = train_df.append(train_reads, ignore_index=True)
        train_df = train_df.sample(frac=1)
        train_df.reset_index(drop=True, inplace=True)
    # train_df = pd.read_csv(args.train_chunks)
    train_df['Index'] = list(range(len(train_df)))
    train_df.set_index('Index')
    # train_df = train_chunks
    train_data = tokenize_chunks(list(train_df['Sequence']), args.k)
    print('Data tokenized')
    if args.vocab != None:
        vocab = load_vocab(args.vocab)
        print(f'Vocab loaded from {args.vocab}')
    else:
        vocab = create_vocab(train_data)
        print(f'Initialized vocab with {len(vocab.stoi)} inputs')
    train_data, slice_lengths = initialize_token_embeddings(vocab, train_data, args.seq_len)

    assert len(train_data) == len(slice_lengths)

    print(f'Token embeddings initialized for {len(train_data)} samples')
    train_df['Tokenized_sequences'] = train_data
    train_df['Slice_lengths'] = slice_lengths
    print(train_df.head())
    print('Added tokenized data to data frame')
    model = RepresentationNetwork(len(vocab.stoi), args.embedding_dim, args.attention_heads, args.embedding_dim,
                                  args.dropout, args.num_layers, devices[0])
    if args.model_checkpoint != None:
        checkpoint = torch.load(args.model_checkpoint)
        model.load_state_dict(checkpoint.state_dict())
        print(f'Checkpoint loaded from {args.model_checkpoint}')
    print('Model created')
    model.to(devices[0])
    model.train()
    print('Model loaded')

    lr = 5e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2500, gamma=0.99)

    save_vocab(vocab, args.model_tgt + 'vocab.txt')

    print('Training beginning')
    for epoch in range(args.epochs):
        avg_loss = 0.0
        for i in tqdm(range(0, len(train_data), args.batch_size)):
            optimizer.zero_grad()

            anchors, positives, negatives, anchors_lens, positives_lens, negatives_lens, anchor_slices, positive_slices, negative_slices = get_positives_negatives(range(i, i + args.batch_size), train_df)

            anchors, positives, negatives = anchors.to(devices[0]), positives.to(devices[0]), negatives.to(devices[0])

            anchor_embedding = model.forward(anchors, anchors_lens, anchor_slices)
            positive_embedding = model.forward(positives, positives_lens, positive_slices)
            negative_embedding = model.forward(negatives, negatives_lens, negative_slices)

            loss = model.get_loss(anchor_embedding, positive_embedding, negative_embedding)
            avg_loss += loss

            if i % 100 == 0:
                print(f'Epoch = {epoch}, step = {i}, loss = {loss}, lr = {scheduler.get_lr()}')

            loss.backward()
            optimizer.step()
            scheduler.step()
            del loss
            del anchor_embedding
            del positive_embedding
            del negative_embedding
            del anchors
            del positives
            del negatives
            del anchors_lens
            del positives_lens
            del negatives_lens
            del anchor_slices
            del positive_slices
            del negative_slices

        print(f'For epoch number {epoch} average loss was {avg_loss / (math.ceil(len(train_data) / args.batch_size))}')
        if (epoch + 1) % 3 == 0:
            torch.save(model, args.model_tgt + f'ep_{epoch}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default=None)
    parser.add_argument('--vocab', default=None)
    parser.add_argument('--train_chunks', default='new_sampling_chunks_30.csv')
    parser.add_argument('--train_reads', default=None)
    parser.add_argument('--k', default=9, type=int)
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gpu', type=str, nargs='+', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model_tgt', default=None)
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