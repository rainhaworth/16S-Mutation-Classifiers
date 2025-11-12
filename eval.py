# train and pickle all models for downstream classification tasks
import numpy as np
import pickle
import argparse
import os
import csv

from utils import print_time, load_data, load_metadata, get_univ_kmers, make_gapped_dataset, make_fn_dataset, find_universal_primers, proportion_kmers_present, clf_eval


###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=float, default=0.1)
    parser.add_argument('-f', type=str, default='./SILVA_138.2_SSURef_NR99_tax_silva_filtered.fasta')
    parser.add_argument('-n', type=int, default=-1)
    parser.add_argument('-k', type=int, default=11)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    mut_rate = args.r

    # load
    with print_time('loading data'):
        gt, mut = load_data(args)
        total_seqs = len(gt)
        print(total_seqs, 'sequences found')

        kms = get_univ_kmers(args)
    
    # gapped kmers
    with print_time('finding gapped kmers'):
        X, y = make_gapped_dataset(gt, mut, kms)
    
    with print_time('gapped kmers'):
        X_gap = np.sum(X, axis=-1)
        print('\t'.join(str(x) for x in clf_eval('./gapped-sum-LR-' + str(mut_rate) +'.pkl', X_gap[:,None], y)))

    # universal primers
    with print_time('universal primers'):
        X, y = make_fn_dataset(gt, mut, find_universal_primers)
        X_up = X
        print('\t'.join(str(x) for x in clf_eval('./primer-LR-' + str(mut_rate) + '.pkl', X[:,None], y)))
    
    # biomarker LR
    with print_time('contiguous kmer biomarkers'):
        # load biomarkers
        with open('./biomarker-kmers.pkl', 'rb') as f:
            kmers_of_interest = pickle.load(f)

        def prop_koi(x):
            return proportion_kmers_present(x, kmers_of_interest)
        
        X, y = make_fn_dataset(gt, mut, prop_koi)
        X_bm = X
        print('\t'.join(str(x) for x in clf_eval('./biomarker-LR-' + str(mut_rate) + '.pkl', X[:,None], y)))
    

    # consensus, if present
    X = np.stack([X_gap, X_up, X_bm], axis=-1)
    with print_time('consensus'):
        cpth_pre = './consensus-'
        cpth_suf = str(mut_rate) + '.pkl'
        cpth_mids = ['max3-', 'max5-', 'max10-', '']
        for cpm in cpth_mids:
            cpth = cpth_pre + cpm + cpth_suf
            if os.path.exists(cpth):
                print('\t'.join(str(x) for x in clf_eval(cpth, X, y)))

    if args.save:
        with print_time('saving datasets'):
            mds = load_metadata(args)
            dir = os.path.dirname(args.f)
            fn1 = os.path.join(dir, 'data-gt.csv')
            fn2 = os.path.join(dir, 'data-m' + str(args.r) + '.csv')
            with open(fn1, 'w') as f1, open(fn2, 'w') as f2:
                w1 = csv.writer(f1)
                w2 = csv.writer(f2)
                header = ['description', 'gapped', 'primers', 'frequent']
                w1.writerow(header)
                w2.writerow(header)
                for i, md in enumerate(mds):
                    row1 = [md] + [str(x) for x in X[i*2]]
                    row2 = [md] + [str(x) for x in X[i*2+1]]
                    w1.writerow(row1)
                    w2.writerow(row2)