import argparse
from Bio import SeqIO
import random
import pandas as pd


class Read:
    def __init__(self, read, tgt_sequence, general_ref, start_position):
        self.sequence = read
        self.start_position = start_position
        self.length = len(read)
        self.reference_label = tgt_sequence
        self.general_ref = general_ref

    def _to_dict(self):
        return {'Sequence': self.sequence,
            'Start position': self.start_position,
            'Length': self.length,
            'Reference': self.reference_label,
            'General_ref': self.general_ref
                }


def sample_reads(ref_dirs, mappings, num):
    """
    We sample reads using information from mapping; we sample reads mapped to + strand with more than 80% of read
    mapped
    :param ref_dirs: directories with reads
    :param num: number of sampled to be sampled
    :return:
    """
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
    reads = []
    per_ref = num // len(ref_dirs)
    for ref_dir, mapping in zip(ref_dirs, mappings):
        print(ref_dir, mapping)
        reference_name = ref_dir.split('/')[-2]
        maps = load_mappings(mapping)
        read_seqs = SeqIO.parse(ref_dir + 'reads.fastq', 'fastq')
        seqs = dict([(r.id, r.seq) for r in read_seqs])
        maps['Read'] = maps[0].apply(lambda x: str(seqs[x]))
        maps['Lens'] = maps['Read'].apply(lambda x: len(x))
        print(maps.head())
        print(len(maps))
        maps = maps.loc[((maps[3] - maps[2]) > 0.8 * maps[1]) & (maps[4] == '+')] #& (maps['Lens'] > 2*4608)]
        print(f'Loaded {len(maps)} reads for reference {new_to_old[reference_name]}')
        assert len(seqs) >= per_ref
        sampled_reads = maps.sample(n=per_ref, replace=False)
        choices = [Read(row['Read'], row[5], new_to_old[reference_name], row[7]) for i, row in sampled_reads.iterrows()]
        #choices = [Read(r_seq, list(maps[maps[0] == r_id][5])[0], new_to_old[reference_name], list(maps[maps[0] == r_id][7])[0]) for r_id, r_seq in sampled_reads]
        """while i < per_ref:
            read_id, read_sequence = random.choice(seqs)
            mapping_info = maps[maps[0] == read_id]
            mapping_info = mapping_info.loc[((mapping_info[3] - mapping_info[2]) > 0.8 * mapping_info[1]) & (mapping_info[4] == '+')]
            if len(mapping_info) == 1:
                choices.append(Read(read_sequence, list(mapping_info[5])[0], new_to_old[reference_name], list(mapping_info[7])[0]))"""
        reads.extend(choices)
    random.shuffle(reads)
    return reads

def save_reads(chunks, tgt_file):
    df = pd.DataFrame.from_records([chunk._to_dict() for chunk in chunks])
    df.to_csv(tgt_file)

def load_mappings(mapping_file):
    mapping_df = pd.read_csv(mapping_file, header=None, sep='\t')
    return mapping_df


def main(args):
    reads = sample_reads(args.references, args.mappings, args.num_chunks)
    save_reads(reads, args.tgt_chunk_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', nargs='+')
    parser.add_argument('--mappings', nargs='+')
    parser.add_argument('--num_chunks', type=int, default=10000)
    parser.add_argument('--tgt_chunk_file')
    args = parser.parse_args()

    main(args)