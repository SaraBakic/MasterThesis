import argparse
import pandas as pd
from gensim.models import Word2Vec
import math


def tokenize_chunks(chunks, k):
    tokenized_chunks = []
    for i, chunk in enumerate(chunks):
        if type(chunk) == float and math.isnan(chunk):
            print(f"Chunk number {i} is nan")
            continue
        tokenized_chunk = []
        for i in range(0, len(chunk), k):
            tokenized_chunk.append(chunk[i:i+k])
        tokenized_chunks.append(tokenized_chunk)
    return tokenized_chunks


def main(args):
    chunk_df = pd.read_csv(args.chunks_path)
    tokenized_chunks = tokenize_chunks(list(chunk_df['Sequence']), args.k)
    cbow_model = Word2Vec(tokenized_chunks, min_count=1, vector_size=64, window=5, workers=4)
    cbow_model.save(args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks_path')
    parser.add_argument('--k', type=int)
    parser.add_argument('--model_path')
    args = parser.parse_args()
    main(args)