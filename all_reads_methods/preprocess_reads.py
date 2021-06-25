import argparse
from Bio import SeqIO
import random
import pandas as pd
import os


class Chunk:
    def __init__(self, sequence, start_position, length, reference_label, general_ref):
        self.sequence = sequence
        self.start_position = start_position
        self.length = length
        self.reference_label = reference_label
        self.general_ref = general_ref

    def __repr__(self):
        return f'{self.sequence}\t{self.start_position}\t{self.length}\t{self.reference_label}\t{self.general_ref}'

    def _to_dict(self):
        return {
            'Sequence': self.sequence,
            'Start position': self.start_position,
            'Length': self.length,
            'Reference': self.reference_label,
            'General_ref': self.general_ref
        }


def sample_chunks(references, ref_labels, chunk_size, num_chunks):
    chunks = []
    i = 0
    while i < num_chunks:
        genome_index = random.randint(0, len(references) - 1)
        genome = references[genome_index]
        weights = [len(r_seq) for r_id, r_seq in genome]
        reference_index = random.choices(range(len(genome)), weights=weights, k=1)[0]
        ref_id, reference = genome[reference_index]
        general_label = ref_labels[genome_index]
        start_position = random.randint(0, len(reference))
        chunk = reference[start_position:start_position+chunk_size]
        if len(chunk) < chunk_size:
            print(f"Skipping chunk {chunk}")
            continue
        chunks.append(Chunk(chunk, start_position, chunk_size, ref_id, general_label))
        i += 1

    return chunks

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


def save_chunks(chunks, tgt_file):
    df = pd.DataFrame.from_records([chunk._to_dict() for chunk in chunks])
    df.to_csv(tgt_file, index=False)


def main(args):
    references, ref_labels = load_references(args.references)
    chunks = sample_chunks(references, ref_labels, args.chunk_size, args.num_chunks)
    save_chunks(chunks, args.tgt_chunk_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', nargs='+')
    parser.add_argument('--chunk_size', type=int, default=9216)
    parser.add_argument('--num_chunks', type=int, default=100000)
    parser.add_argument('--tgt_chunk_file')
    args = parser.parse_args()

    main(args)