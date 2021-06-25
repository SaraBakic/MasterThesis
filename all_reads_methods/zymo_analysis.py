import argparse
from Bio import SeqIO
import numpy as np
import pandas as pd
import cigar
import matplotlib.pyplot as plt
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--reads', nargs='+')
parser.add_argument('--type', choices=['original', 'rle', 'soft_clipped', 'hard_clipped', 'soft_full_ratio'])
parser.add_argument('--remove_outliers', action='store_true')
parser.add_argument('--tgt_dir')
parser.add_argument('--mappings', default=None)
args = parser.parse_args()


def load_og_lens(files):
    lens = []
    for file in files:
        reads = SeqIO.parse(file, 'fastq')
        lens.extend([len(record.seq) for record in reads])
        print(f'Parsed {file}')
    return np.array(lens)


def load_rle_lens(file):
    with open(file, 'r') as reads_file:
        lens = [len(line.strip()) for line in reads_file]
    return np.array(lens)


def load_paf_lens(file, type):
    cigars = extract_cigar(file)
    clipping_tag = 'S' if type == 'soft_clipped' else 'H'
    relevant_tags = [list(filter(lambda x: x[1] == clipping_tag, cigar)) for cigar in cigars]
    relevants_lens = [[c[0] for c in cigar] for cigar in relevant_tags]
    total_lens = [sum(l) for l in relevants_lens]
    return np.array(total_lens)


def extract_cigar(file):    #TODO write sam parsing
    #mapping_df = pd.read_csv(file, header=9, sep='\t')
    cigars = []
    with open(file, 'r') as sam_file:
        for line in sam_file:
            line = line.strip().split('\t')
            if len(line) < 21:
                continue
            cigars.append(line[5])
    cigars = [list((cigar.Cigar(c)).items()) for c in cigars]
    return cigars


def soft_full_ratio(mappings, reads):
    ratios = []
    read_info = {}
    reads = SeqIO.parse(reads, 'fastq')
    for read in reads:
        read_info[read.name] = len(read.seq)

    with open(mappings, 'r') as sam_file:
        for line in sam_file:
            line = line.strip().split('\t')
            if len(line) < 21:
                continue
            read = line[0]
            cig = list(cigar.Cigar(line[5]).items())
            cig = list(filter(lambda x: x[1] == 'S', cig))
            soft_len = sum([len for (len, sign) in cig])
            ratios.append(soft_len/read_info[read])

    return ratios


def remove_outliers(lens):
    lens = sorted(lens)
    outliers = int(0.05*len(lens))
    lens = lens[outliers:(len(lens)-outliers)]
    return lens


def plot_hist(lens, type):
    median = statistics.median(lens)
    print(f'Median value is {median}')
    if args.remove_outliers:
        lens = remove_outliers(lens)
    print(min(lens), max(lens))
    num_bins = 50
    n, bins, patches = plt.hist(lens, num_bins, facecolor='cornflowerblue')
    plt.axvline(median, color='k', linestyle='dashed', linewidth=1)
    plt.show()
    outliers_removed = 'no outliers' if args.remove_outliers else ''
    plt.title(f'Histogram for {type} reads {outliers_removed}')
    plt.savefig(args.tgt_dir + f'hist_{type}{"_".join(outliers_removed)}.png')
    plt.close()


def main():
    if args.type == 'original':
        lens = load_og_lens(args.reads)
    elif args.type == 'rle':
        lens = load_rle_lens(args.reads)
    elif args.type == 'soft_full_ratio':
        lens = soft_full_ratio(args.mappings, args.reads)
    else:
        lens = load_paf_lens(args.reads, args.type)
    for k in [5, 7, 9, 11]:
        for s in [128, 256, 512, 1024]:
            print(f'{sum(lens<=(k*s))/len(lens)*100}% of reads has length less than {k}*{s}={k*s}')
    plot_hist(lens, args.type)


if __name__ == '__main__':
    main()