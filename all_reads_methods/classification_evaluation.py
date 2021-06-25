import argparse
import os
from vocab_utils import load_vocab
from representation_evaluation import rename_layers
from classification_learning import references_to_labels, get_labels
from initial_kmer_embeddings import tokenize_chunks
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


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


def evaluate(model, eval_data, eval_labels, eval_lens, batch_size, device, tgt_path, epoch=0):
    predictions = []
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            eval_batch = torch.cat(eval_data[i:i+batch_size]).to(device)
            lens = eval_lens[i:i+batch_size]
            logits = model.forward(eval_batch, lens)
            print(logits)
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


def read_ref_labels(labels_path):
    old_to_new = {'Listeria_monocytogenes_complete_genome': 'lm',
                  'Salmonella_enterica_complete_genome': 'se',
                  'Staphylococcus_aureus_complete_genome': 'sa',
                  'Cryptococcus_neoformans_draft_genome': 'cn',
                  'Escherichia_coli_complete_genome': 'ec',
                  'Pseudomonas_aeruginosa_complete_genome': 'pa',
                  'Lactobacillus_fermentum_complete_genome': 'lf',
                  'Enterococcus_faecalis_complete_genome': 'ef',
                  'Bacillus_subtilis_complete_genome': 'bs',
                  'Saccharomyces_cerevisiae_draft_genome': 'sc'
                  }
    new_to_old = {'lm': 'Listeria_monocytogenes_complete_genome',
                  'se': 'Salmonella_enterica_complete_genome',
                  'sa': 'Staphylococcus_aureus_complete_genome',
                  'cn': 'Cryptococcus_neoformans_draft_genome',
                  'ec': 'Escherichia_coli_complete_genome',
                  'pa': 'Pseudomonas_aeruginosa_complete_genome',
                  'lf': 'Lactobacillus_fermentum_complete_genome',
                  'ef': 'Enterococcus_faecalis_complete_genome',
                  'bs': 'Bacillus_subtilis_complete_genome',
                  'sc': 'Saccharomyces_cerevisiae_draft_genome'

    }
    seq_to_label = {}
    with open(labels_path, 'r') as labels_file:
        for l in labels_file:
            key, label = l.strip().split(': ')
            label = int(label)
            seq_to_label[key] = label
    return seq_to_label


def main(args, devices):
    embedding_model = torch.load(args.model_path)
    vocab = load_vocab(args.vocab_path)
    embedding_model.to(devices[0])
    embedding_model.eval()

    eval_df = pd.read_csv(args.chunks)
    eval_data = tokenize_chunks(list(eval_df['Sequence']), 9)
    eval_lens = [len(d) if len(d) < 512 else 512 for d in eval_data]
    eval_data = initialize_token_embeddings(vocab, eval_data, args.seq_len)

    #seq_to_label, label_to_seq = references_to_labels(list(eval_df['Reference']))
    seq_to_label = read_ref_labels(args.reference_labels)
    print(seq_to_label)

    eval_labels = get_labels(list(eval_df['General_ref']), seq_to_label)
    evaluate(embedding_model, eval_data, eval_labels, eval_lens, args.batch_size, devices[0], f'{args.tgt_dir}{args.type}_reads_')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks')
    parser.add_argument('--model_path')
    parser.add_argument('--vocab_path')
    parser.add_argument('--reference_labels')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--type', default='longer')
    parser.add_argument('--tgt_dir')
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
