def save_vocab(vocab, path):
    with open(path, 'w') as f:
        for token, index in vocab.items():
            f.write(f'{token}\t{index}')


def load_vocab(path):
    vocab = {}
    with open(path, 'r') as f:
        for l in f:
            token, index = l.strip().split('\t')
            vocab[token] = int(index)
    return vocab