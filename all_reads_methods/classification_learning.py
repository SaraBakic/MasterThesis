import os
import argparse
import torch
import pandas as pd
import numpy as np
import math
from classification_network import ClassificationNetwork
from representation_network import RepresentationNetwork
from tqdm import tqdm
from vocab_utils import load_vocab
from initial_kmer_embeddings import tokenize_chunks
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


"""def initialize_token_embeddings(vocab, tokenized_sequences):
    data = [torch.unsqueeze(torch.tensor([vocab[kmer] if (kmer in vocab.keys()) else vocab['<unk>'] for kmer in seq], dtype=torch.long), 0) for seq in tokenized_sequences]
    #return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    return data"""


def initialize_token_embeddings(vocab, tokenized_sequences, seq_len):
    """
    :return: Returns a list of long tensors where each tensor represents a chunk and each value in the tensor is vocab
    index of a "chunk token"
    """
    #data = [torch.unsqueeze(torch.tensor([vocab[kmer] if (kmer in vocab.keys()) else vocab['<unk>'] for kmer in seq], dtype=torch.long), 0) for seq in tokenized_sequences]
    data = []
    for seq in tokenized_sequences:
        t = torch.unsqueeze(torch.tensor([vocab[kmer] if (kmer in vocab.keys()) else vocab['<unk>'] for kmer in seq], dtype=torch.long), 0)
        if t.size()[1] > seq_len:
            data.append(t.narrow(1, 0, seq_len))
        elif t.size()[1] < seq_len:
            padding = torch.tensor([[vocab['<pad>']]*(seq_len - t.size()[1])], dtype=torch.long)
            data.append(torch.cat((t, padding), dim=1))
        else:
            data.append(t)
    #return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    return data


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


def evaluate(model, eval_data, eval_labels, eval_lens, batch_size, device, tgt_path, epoch):
    predictions = []
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            eval_batch = torch.cat(eval_data[i:i+batch_size]).to(device)
            lens = eval_lens[i:i+batch_size]
            logits = model.forward(eval_batch, lens)
            predicted = torch.argmax(logits, dim=1).to('cpu').tolist()
            predictions.extend(predicted)
    conf_matrix = confusion_matrix(eval_labels, predictions)
    accuracy = accuracy_score(eval_labels, predictions)
    recall = recall_score(eval_labels, predictions, average='macro')
    precision = precision_score(eval_labels, predictions, average='macro')
    f1 = f1_score(eval_labels, predictions, average='macro')
    with open(tgt_path + 'evaluation_log.txt', 'a') as log_file:
        log_file.write(f'Evaluation results afer epoch {epoch}: accuracy = {accuracy}, macro recall = {recall}, macro precision = {precision}, macro F1 = {f1}, confusion matrix = \n')
        log_file.write(f'{np.array2string(conf_matrix, separator=",")}\n')
    #print(f'Evalutation results: accuracy = {accuracy}, recall = {recall}, precision = {precision}, micro F1 = {f1}, confusion matrix = ')
    #print(conf_matrix)


def main(args, devices):
    embedding_model_checkpoint = torch.load(args.model_path)
    vocab = load_vocab(args.vocab_path)
    embedding_model = RepresentationNetwork(len(vocab), args.embedding_dim, 8, args.embedding_dim, 0.1, 6, devices[0])
    embedding_model.load_state_dict(embedding_model_checkpoint.state_dict())

    train_chunks = pd.read_csv(args.train_chunks)
    #train_df = pd.read_csv(args.train_chunks)
    if args.train_reads != None:
        train_reads = pd.read_csv(args.train_reads)
        print(f'Loaded {len(train_chunks)} chunks samples and {len(train_reads)} read samples')
        frames = [train_reads, train_chunks]
        train_df = pd.concat(frames)
        del train_chunks
        del train_reads
        train_df.reset_index(drop=True, inplace=True)
        train_df = train_df.sample(frac=1)
        train_df.reset_index(drop=True)
        train_df['Index'] = list(range(len(train_df)))
        train_df.set_index('Index')
    else:
        train_df = train_chunks
    train_data = tokenize_chunks(list(train_df['Sequence']), args.k)
    train_lens = [len(d) if len(d) < args.seq_len else args.seq_len for d in train_data]
    train_data = initialize_token_embeddings(vocab, train_data, args.seq_len)
    print(train_lens[:20])

    eval_df = pd.read_csv(args.eval_chunks)
    eval_data = tokenize_chunks(list(eval_df['Sequence']), args.k)
    eval_lens = [len(d) if len(d) < 512 else 512 for d in eval_data]
    eval_data = initialize_token_embeddings(vocab, eval_data, args.seq_len)

    seq_to_label, label_to_seq = references_to_labels(list(train_df['General_ref']))

    with open(args.model_tgt + 'sequence_labels.txt', 'w') as seq_lab_file:
        for key, val in seq_to_label.items():
            seq_lab_file.write(f'{key}: {val}\n')

    train_labels = get_labels(list(train_df['General_ref']), seq_to_label)
    eval_labels = get_labels(list(eval_df['General_ref']), seq_to_label)

    classification_model = ClassificationNetwork(embedding_model, args.embedding_dim, len(seq_to_label), freeze_embeddings=args.freeze_embeddings)
    classification_model.to(devices[0])
    classification_model.train()

    lr = args.lr
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classification_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5000, gamma=0.99)


    for epoch in range(args.epochs):
        avg_loss = 0.0
        for i in tqdm(range(0, len(train_data), args.batch_size)):
            optimizer.zero_grad()

            batch = torch.cat(train_data[i:i+args.batch_size]).to(devices[0])
            lens = train_lens[i:i+args.batch_size]
            batch_labels = train_labels[i:i+args.batch_size].to(devices[0])
            logits = classification_model.forward(batch, lens)

            loss = loss_func(logits, batch_labels)
            avg_loss += loss

            if i % 12000 == 0:
                print(f'Epoch = {epoch}, step = {i}, loss = {loss}, lr = {scheduler.get_lr()}')

            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f'For epoch number {epoch} average loss was {avg_loss/(math.ceil(len(train_data)/args.batch_size))}')

        print("Starting evaluation")
        evaluate(classification_model, eval_data, eval_labels, eval_lens, args.batch_size, devices[0], args.model_tgt, epoch)

        if epoch % 5 == 0:
            torch.save(classification_model, args.model_tgt + f'ep_{epoch}.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--vocab_path')
    parser.add_argument('--train_chunks')
    parser.add_argument('--train_reads')
    parser.add_argument('--eval_chunks')
    parser.add_argument('--model_tgt')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--freeze_embeddings', action='store_true', default=False)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--k', type=int, default=9)
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--lr', type=float, default=8e-3)
    parser.add_argument('--gpu', type=str, nargs='+')
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