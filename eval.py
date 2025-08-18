# train and pickle all models for downstream classification tasks
from sklearn import tree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
from Bio import SeqIO
from ast import literal_eval
from collections import defaultdict, Counter
import re
import random
import pickle
import time
import argparse

# globals
mut_rate = 0.1
k = 11

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
def load_data(fn, max_seqs = -1):
    seqs_gt = []
    seqs_mut = []
    for h,i in enumerate(SeqIO.parse(fn,'fasta')):
        s,L = str(i.seq).upper().replace('T','U'),len(i.seq)
        mutated_seq = mutate_dna(s, int(mut_rate*L))
        seqs_gt.append(s)
        seqs_mut.append(mutated_seq)
        if h == max_seqs-1: break
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

# evaluate from y_pred
def pred_eval(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return tn, fp, fn, tp, tn/(tn+fn), tp/(tp+fp)

# evaluate from classifier
def clf_eval(clf, X, y):
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

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='./SILVA_138.2_SSURef_NR99_tax_silva_filtered.fasta')
    parser.add_argument('-m', '--max_seqs', type=int, default=-1)
    args = parser.parse_args()

    # load
    with print_time('loading data'):
        gt, mut = load_data(args.f, args.max_seqs)
        total_seqs = len(gt)
        print(total_seqs, 'sequences found')
    
    # gapped kmers
    with print_time('finding gapped kmers'):
        X, y = make_gapped_dataset(gt, mut)
        
    # with thresholding
    with print_time('gap simple'):
        DB = 30 # hardcode after manual tuning
        X_sum = np.sum(X, axis=-1)
        y_pred = X_sum >= DB
        print(pred_eval(y, y_pred))

    with print_time('gap high freq'):
        # get kmer frequencies on training set
        fn = './results/k-' + str(k) + '-300000.csv'
        kmer_counts = Counter()
        for i, line in enumerate(open(fn, 'r')):
            kmer_counts[i] = int(line.split(';')[1])
        kmer_freqs = [(x,y/total_seqs) for x,y in kmer_counts.most_common()]

        # high frequency kmers
        kmer_subset = [i for i in range(len(kmer_freqs)) if kmer_freqs[i][1] > 0.5]
        X_subset = X[:,kmer_subset]

        # hardcode threshold again
        DB = 26
        y_pred = np.sum(X_subset, axis=-1) >= DB
        print(pred_eval(y, y_pred))
    
    with print_time('gap joint'):
        with open('./jc-biomarkers.pkl', 'rb') as f:
            biomarkers = pickle.load(f)

        biomarker_keys = set(biomarkers.keys())

        # count biomarkers
        n_marker = np.zeros_like(y, dtype=int)
        for i in range(len(n_marker)):
            km_ids = np.nonzero(X[i])[0]
            for j in range(len(km_ids)):
                if j not in biomarker_keys: continue
                for kmi in km_ids[j+1:]:
                    if kmi in biomarkers[j]:
                        n_marker[i] += 1
        
        # hardcode another threshold
        DB = 343
        y_pred = n_marker >= DB
        print(pred_eval(y, y_pred))

    # with ML
    with print_time('gap DT'):
        with open('./gapped-DT.pkl', 'rb') as f:
            clf = pickle.load(f)
            print(clf_eval(clf, X, y))

    with print_time('gap GB'):
        with open('./gapped-GB.pkl', 'rb') as f:
            clf = pickle.load(f)
            print(clf_eval(clf, X, y))

    with print_time('gap LR'):
        with open('./gapped-LR.pkl', 'rb') as f:
            clf = pickle.load(f)
            print(clf_eval(clf, X, y))
    
    # homopolymer LR
    with print_time('homopolymer LR'):
        # make dataset
        X = np.zeros(len(gt)*2)
        y = np.zeros(len(gt)*2, int)

        for i in range(len(gt)):
            hp_gt = find_homopolymer(gt[i])
            hp_mut = find_homopolymer(mut[i])

            # avoid division by 0
            gt_s = sum(hp_gt)
            mut_s = sum(hp_mut)
            if gt_s == 0: gt_s = 1
            if mut_s == 0: mut_s = 1

            # compute G homopolymer bias for single feature
            X[i*2] = (hp_gt[2] / gt_s)
            y[i*2] = 1

            X[i*2+1] = (hp_mut[2] / mut_s)
            y[i*2+1] = 0
        
        # load classifier
        with open('./homopolymer-LR.pkl', 'rb') as f:
            clf = pickle.load(f)
            print(clf_eval(clf, X[:,None], y))

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
        
        # load classifier
        with open('./primer-LR.pkl', 'rb') as f:
            clf = pickle.load(f)
            print(clf_eval(clf, X[:,None], y))
    
    # biomarker LR
    with print_time('biomarker LR'):
        # load biomarkers
        with open('./biomarker-kmers.pkl', 'rb') as f:
            kmers_of_interest = pickle.load(f)

        # make dataset
        X = np.zeros(len(gt)*2)
        y = np.zeros(len(gt)*2, int)

        for i in range(len(gt)):
            X[i*2] = proportion_kmers_present(gt[i], kmers_of_interest)
            y[i*2] = 1

            X[i*2+1] = proportion_kmers_present(mut[i], kmers_of_interest)
            y[i*2+1] = 0
        
        # load classifier
        with open('./biomarker-LR.pkl', 'rb') as f:
            clf = pickle.load(f)
            print(clf_eval(clf, X[:,None], y))