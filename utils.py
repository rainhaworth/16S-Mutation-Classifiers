import random
import pickle
import re
import time
import numpy as np
from Bio import SeqIO
from collections import defaultdict
from ast import literal_eval
from sklearn.metrics import confusion_matrix

# assume args contains: r = mutation rate, f = fasta file, n = # seqs, k = kmer length

# function definitions (from notebook)

# apply a fixed number of mutations
def mutate_rna(rna_string, num_mutations):
    rna_list = list(rna_string)
    rna_length = len(rna_list)
    if num_mutations > rna_length:
        raise ValueError("Number of mutations cannot exceed the length of the RNA string.")

    mutation_indices = random.sample(range(rna_length), num_mutations)

    for index in mutation_indices:
        current_base = rna_list[index]
        valid_bases = ['A', 'U', 'C', 'G']
        if current_base not in valid_bases:
            continue
        else:
            valid_bases.remove(current_base) # Ensure the new base is different.
            new_base = random.choice(valid_bases)
            rna_list[index] = new_base

    return "".join(rna_list)  # Convert back to a string

# load sequences + create mutated sequences
def load_data(args):
    total_seqs = args.n
    fasta_f = args.f
    seqs_gt = []
    seqs_mut = []
    for h,i in enumerate(SeqIO.parse(fasta_f,'fasta')):
        if h == total_seqs: break
        s,L = str(i.seq).upper(),len(i.seq)
        mutated_seq = mutate_rna(s, int(args.r*L))
        seqs_gt.append(s)
        seqs_mut.append(mutated_seq)
    return seqs_gt, seqs_mut

# get metadata in same order
def load_metadata(args):
    total_seqs = args.n
    fasta_f = args.f
    md = []
    for h,i in enumerate(SeqIO.parse(fasta_f,'fasta')):
        if h == total_seqs: break
        md.append(str(i.description))
    return md

# load multiple datasets split by taxonomy
def load_data_by_taxon(args):
    seqs_gt = defaultdict(list)
    seqs_mut = defaultdict(list)
    for h,i in enumerate(SeqIO.parse(args.f,'fasta')):
        if h == args.n: break
        s,d,L = str(i.seq).upper(),i.description,len(i.seq)
        mutated_seq = mutate_rna(s, int(args.r*L))

        taxon = d.split(' ')[1].split(';')
        taxon = ';'.join(taxon[:args.t+1])

        seqs_gt[taxon].append(s)
        seqs_mut[taxon].append(mutated_seq)
    return seqs_gt, seqs_mut

# construct kmers from universally conserved nucleotides; could add these extra bits to argparse but they aren't going to change
def get_univ_kmers(args, fn='./universal-residues-ecoli.txt', maxsize=100):
    k = args.k

    with open(fn, 'r') as f:
        residues = f.readlines()
    residues = [(x[0], int(x[1:])) for x in residues]

    # compute relative offsets p_{i+j} - p_i, reject gapped kmers with total width above maxsize
    kmers = [tuple((residues[i+j][0], residues[i+j][1] - residues[i][1]) for j in range(k))
             for i in range(len(residues)-k+1)
             if residues[i+k-1][1] - residues[i][1] <= maxsize]
    
    return kmers

def make_kmer_regexes(kmers):
    k = len(kmers[0])
    kmer_regexes = []
    for km in kmers:
        re_str = ''
        chars = [x[0] for x in km]
        offsets = [x[1] for x in km]
        rel_offs = [offsets[i] - offsets[i-1] for i in range(1,k)]
        for i in range(k):
            re_str += chars[i]
            if i < k-1 and rel_offs[i] > 1:
                re_str += '.{' + str(rel_offs[i]-1) + '}'
        kmer_regexes.append(re.compile(re_str))
    return kmer_regexes

# make gapped kmer dataset for sklearn
def make_gapped_dataset(seqs_gt, seqs_mut, kms):
    krs = make_kmer_regexes([kms[i] for i in range(len(kms))])

    X = np.zeros((len(seqs_gt)*2, len(kms)), int)
    y = np.zeros(len(seqs_gt)*2, int)

    for i in range(len(seqs_gt)):
        s = seqs_gt[i]
        mutated_seq = seqs_mut[i]

        X[i*2] = [kr.search(s) != None for kr in krs]
        y[i*2] = 1

        X[i*2+1] = [kr.search(mutated_seq) != None for kr in krs]
        y[i*2+1] = 0

    return X, y

# make dataset by applying any function to each seq_gt, seq_mut
def make_fn_dataset(seqs_gt, seqs_mut, fn):
    X = np.zeros(len(seqs_gt)*2, type(fn(seqs_gt[0])))
    y = np.zeros(len(seqs_gt)*2, int)

    for i in range(len(seqs_gt)):
        X[i*2] = fn(seqs_gt[i])
        y[i*2] = 1

        X[i*2+1] = fn(seqs_mut[i])
        y[i*2+1] = 0

    return X, y

# find homopolymers
def find_homopolymer(seq, shortest_homopolymer_len=4):
    L = len(seq)
    homopolymers = [0,0,0,0]
    i = 0
    
    while i < L - shortest_homopolymer_len + 1:
        base = seq[i]
        homopolymer = base
        for j in range(i+1,L):
            if seq[j] == base:
                homopolymer += base
            else:
                break
        i = j
        if len(homopolymer) >= shortest_homopolymer_len:
            NT = list(set(homopolymer))[0]
            if NT == 'A': homopolymers[0] += 1
            elif NT == 'C': homopolymers[1] += 1
            elif NT == 'G': homopolymers[2] += 1
            elif NT == 'U': homopolymers[3] += 1
    
    return homopolymers

# make reusable primer regexes
universal_primers = ['AGAGUUUGAUCCUGGCUCAG','AGAGUUUGAUC[AC]UGGCUCAG','ACUGCUGC[GC][CU]CCCGUAGGAGUCU','GACUCCUACGGGAGGC[AU]GCAG','GUAUUACCGCGGCUGCUGG','GUGCCAGC[AC]GCCGCGGUAA','GGAUUAGAUACCCUGGUA','GGACUAC[ACG][GC]GGGUAUCUAAU','CCGUCAAUUCCUUU[AG]AGUUU','UAAAACU[CU]AAA[GU]GAAUUGACGGG','[CU]AACGAGCGCAACCC','GGGUUGCGCUCGUUG','GGUUACCUUGUUACGACUU','CGGUUACCUUGUUACGACUU']
up_regexes = [re.compile(x) for x in universal_primers]

# Returns the number of universal 16S rRNA primers (from 16S rRNA wikipedia page) found in an RNA string.
def find_universal_primers(s):
    matches = set()
    
    for regex in up_regexes:
        for match in regex.finditer(s):
            matches.add(match[0])
    
    return len(matches)

# Returns proportion of biomarker k-mers of interest that are found.
def proportion_kmers_present(seq, biomarker_kmers):
    present = 0
    for k in biomarker_kmers:
        if k in seq:
            present += 1
    return present/len(biomarker_kmers)

## Function finds biomarker k-mers in the SILVA 16S rRNA database sequences.
## A biomarker k-mer has length klen and is found in some minimum proportion (min_freq_SILVA) of SILVA sequences.
def find_biomarker_kmers(args, min_freq_SILVA=0.5):
    klen = args.k
    fasta_f = args.f
    max_seqs = args.n

    kmer_counts = defaultdict(int)
    kmers_of_interest = set()
    
    seqs = 0
    for i in SeqIO.parse(fasta_f,'fasta'):
        if seqs == max_seqs: break
        s = str(i.seq).upper().replace('T','U')
        kmers = {s[i:i+klen] for i in range(len(s)+1-klen)}
        for k in kmers:
            kmer_counts[k] += 1
        seqs += 1
    
    for k,v in sorted(kmer_counts.items(),key=lambda x:x[1]):
        if v/seqs > min_freq_SILVA:
            kmers_of_interest.add(k)

    return kmers_of_interest

# standard training run logic
def train_save_clf(clf, X, y, name, filename):
    with print_time(name):
        clf.fit(X, y)
        with open(filename, 'wb') as f:
            pickle.dump(clf, f)

# evaluate from y_pred
def pred_eval(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel().tolist()
    return tn, fp, fn, tp, tp/(tp+fn), tn/(fp+tn)

# evaluate from saved classifier
def clf_eval(filename, X, y):
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    y_pred = clf.predict(X)
    return pred_eval(y, y_pred)

# timer
class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')