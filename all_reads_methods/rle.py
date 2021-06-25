import argparse
from Bio import SeqIO

parser = argparse.ArgumentParser()
parser.add_argument('--reads', default='test.fastq')
parser.add_argument('--tgt_dir', default='./')
args = parser.parse_args()

def rle(read):
    encoding = ''
    lens = []
    previous_base = ''
    count = 1

    for base in read:
        if base != previous_base:
            if previous_base:
                encoding += previous_base
                lens.append(count)
            count = 1
            previous_base = base
        else:
            count += 1
            continue
    encoding += previous_base
    lens.append(count)

    return encoding, lens


def main():
    reads = SeqIO.parse(args.reads, 'fastq')
    reads = [record.seq for record in reads]

    encodings_lens = [rle(read) for read in reads]

    with open(args.tgt_dir + 'rle_encodings.txt', 'w') as enc_file, open(args.tgt_dir + 'rle_original_lengths.txt', 'w') as lens_file:
        for encoding, original_lens in encodings_lens:
            enc_file.write(encoding + '\n')
            lens_file.write(str(original_lens) + '\n')


if __name__ == '__main__':
    main()
