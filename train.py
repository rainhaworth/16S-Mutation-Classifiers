# train and pickle all models for downstream classification tasks
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from Bio import SeqIO
from ast import literal_eval
from collections import defaultdict
import re
import random
import pickle
import time

# globals
mut_rate = 0.1
k = 11
total_seqs = 282395
fasta_f = 'SILVA_138.2_SSURef_NR99_tax_silva_filtered.fasta'

# fetch kmers
fn = './results/k-'+str(k)+'-300000.csv'
kms = dict()
with open(fn, 'r') as f:
    for i, line in enumerate(f):
        kms[i] = literal_eval(line.split(';')[0])

# function definitions (from notebook)

def make_kmer_regexes(kmers):
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

# apply a fixed number of mutations
def mutate_dna(dna_string, num_mutations):
    dna_list = list(dna_string)
    dna_length = len(dna_list)
    if num_mutations > dna_length:
        raise ValueError("Number of mutations cannot exceed the length of the DNA string.")

    mutation_indices = random.sample(range(dna_length), num_mutations)

    for index in mutation_indices:
        current_base = dna_list[index]
        valid_bases = ['A', 'U', 'C', 'G']
        if current_base not in valid_bases:
            continue
        else:
            valid_bases.remove(current_base) # Ensure the new base is different.
            new_base = random.choice(valid_bases)
            dna_list[index] = new_base

    return "".join(dna_list)  # Convert back to a string

# load sequences + create mutated sequences
def load_data():
    seqs_gt = [''] * total_seqs
    seqs_mut = [''] * total_seqs
    for h,i in enumerate(SeqIO.parse(fasta_f,'fasta')):
        s,L = str(i.seq).upper(),len(i.seq)
        mutated_seq = mutate_dna(s, int(mut_rate*L))
        seqs_gt[h] = s
        seqs_mut[h] = mutated_seq
        #if h == total_seqs-1: break
    return seqs_gt, seqs_mut

# make gapped kmer dataset for sklearn
def make_gapped_dataset(seqs_gt, seqs_mut):
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
def find_biomarker_kmers(klen, min_freq_SILVA):
    kmer_counts = defaultdict(int)
    kmers_of_interest = set()
    
    for h,i in enumerate(SeqIO.parse(fasta_f,'fasta')):
        s = str(i.seq).upper().replace('T','U')
        kmers = {s[i:i+klen] for i in range(len(s)+1-klen)}
        for k in kmers:
            kmer_counts[k] += 1
        #if h == total_seqs-1: break
    
    for k,v in sorted(kmer_counts.items(),key=lambda x:x[1]):
        if v/total_seqs > min_freq_SILVA:
            kmers_of_interest.add(k)

    return kmers_of_interest

# timer
class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')

###############################################################################

if __name__ == '__main__':
    # load
    with print_time('loading data'):
        gt, mut = load_data()
    
    
    # gapped kmer ML
    with print_time('finding gapped kmers'):
        X, y = make_gapped_dataset(gt, mut)

    with print_time('gap DT'):
        clf = tree.DecisionTreeClassifier()
        clf.fit(X,y)
        with open('gapped-DT.pkl', 'wb') as f:
            pickle.dump(clf, f)

    with print_time('gap GB'):
        clf = HistGradientBoostingClassifier()
        clf.fit(X,y)
        with open('gapped-GB.pkl', 'wb') as f:
            pickle.dump(clf, f)

    with print_time('gap LR'):
        clf = LogisticRegression()
        clf.fit(X,y)
        with open('gapped-LR.pkl', 'wb') as f:
            pickle.dump(clf, f)
    
    # homopolymer LR
    with print_time('homopolymer LR'):
        # make dataset
        X = np.zeros(len(gt)*2)
        y = np.zeros(len(gt)*2, int)

        for i in range(len(gt)):
            hp_gt = find_homopolymer(gt[i])
            hp_mut = find_homopolymer(mut[i])

            # compute G homopolymer bias for single feature
            X[i*2] = hp_gt[2] / sum(hp_gt)
            y[i*2] = 1

            X[i*2+1] = hp_mut[2] / sum(hp_mut)
            y[i*2+1] = 0
        
        # train
        clf = LogisticRegression()
        clf.fit(X[:,None],y)
        with open('homopolymer-LR.pkl', 'wb') as f:
            pickle.dump(clf, f)

    # universal primer LR
    with print_time('universal primer LR'):
        # make dataset
        X = np.zeros(len(gt)*2, int)
        y = np.zeros(len(gt)*2, int)

        for i in range(len(gt)):
            X[i*2] = find_universal_primers(gt[i])
            y[i*2] = 1

            X[i*2+1] = find_universal_primers(mut[i])
            y[i*2+1] = 0
        
        # train
        clf = LogisticRegression()
        clf.fit(X[:,None],y)
        with open('primer-LR.pkl', 'wb') as f:
            pickle.dump(clf, f)
    
    # biomarker LR
    with print_time('finding biomarkers'):
        # find biomarkers
        kmers_of_interest = find_biomarker_kmers(klen = 11, min_freq_SILVA = 0.5)
    
    with print_time('biomarker LR'):
        # make dataset
        X = np.zeros(len(gt)*2)
        y = np.zeros(len(gt)*2, int)

        for i in range(len(gt)):
            X[i*2] = proportion_kmers_present(gt[i], kmers_of_interest)
            y[i*2] = 1

            X[i*2+1] = proportion_kmers_present(mut[i], kmers_of_interest)
            y[i*2+1] = 0
        
        # train
        clf = LogisticRegression()
        clf.fit(X[:,None],y)
        with open('biomarker-LR.pkl', 'wb') as f:
            pickle.dump(clf, f)

        with open('biomarker-kmers.pkl', 'wb') as f:
            pickle.dump(kmers_of_interest, f)