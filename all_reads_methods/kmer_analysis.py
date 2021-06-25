import argparse
from Bio import SeqIO

parser = argparse.ArgumentParser()
parser.add_argument('--reference')
parser.add_argument('--k', type=int)
parser.add_argument('--tgt_dir')
args = parser.parse_args()


def main():
    reads = SeqIO.parse(args.reference, 'fasta')
    reads = [r.seq for r in reads]
    kmers = {}
    for read in reads:
        for i in range(0, len(read) - args.k + 1):
            kmer = read[i:i+args.k]
            if kmer not in kmers:
                kmers[kmer] = 0
            kmers[kmer] += 1

    with open(args.tgt_dir + f'{args.k}_mer_analysis.txt', 'w') as out_file:
        out_file.write(f'Total of {len(kmers.keys())} {args.k}-mers found with following distribution\n')
        for key, val in kmers.items():
            out_file.write(f'{key}: {val}\n')


if __name__ == '__main__':
    main()