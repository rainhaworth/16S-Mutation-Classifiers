# train and pickle all models for downstream classification tasks
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import argparse

from utils import print_time, load_data, get_univ_kmers, make_gapped_dataset, make_fn_dataset, find_universal_primers, find_biomarker_kmers, proportion_kmers_present, train_save_clf

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=float, default=0.1)
    parser.add_argument('-f', type=str, default='./SILVA_138.2_SSURef_NR99_tax_silva_filtered.fasta')
    parser.add_argument('-n', type=int, default=-1)
    parser.add_argument('-k', type=int, default=11)
    args = parser.parse_args()

    # load
    with print_time('loading data'):
        gt, mut = load_data(args)
        kms = get_univ_kmers(args)
    
    # gapped kmer
    with print_time('gapped kmers'):
        X_gap, y = make_gapped_dataset(gt, mut, kms)
        X_gap = np.sum(X_gap, axis=-1)
        train_save_clf(LogisticRegression(), X_gap[:,None], y, 'gap sum LR', 'gapped-sum-LR-' + str(args.r) + '.pkl')
    
    # universal primers
    with print_time('universal primer LR'):
        # make dataset
        X, y = make_fn_dataset(gt, mut, find_universal_primers)
        # train
        train_save_clf(LogisticRegression(), X[:,None], y, 'train', 'primer-LR-' + str(args.r) + '.pkl')

        X_up = X
    
    # biomarker LR
    with print_time('finding biomarkers'):
        # find biomarkers
        kmers_of_interest = find_biomarker_kmers(args)
    with print_time('biomarker LR'):
        # make dataset
        def prop_koi(x):
            return proportion_kmers_present(x, kmers_of_interest)
        X, y = make_fn_dataset(gt, mut, prop_koi)
        # train
        train_save_clf(LogisticRegression(), X[:,None], y, 'train', 'biomarker-LR-' + str(args.r) + '.pkl')
        
        X_bm = X

    with open('biomarker-kmers.pkl', 'wb') as f:
        pickle.dump(kmers_of_interest, f)

    # make full dataset + train decision trees of various depths
    X = np.stack([X_gap, X_up, X_bm], axis=-1)
    train_save_clf(tree.DecisionTreeClassifier(max_depth=3), X, y, 'training max 3', 'consensus-max3-' + str(args.r) + '.pkl')
    train_save_clf(tree.DecisionTreeClassifier(max_depth=5), X, y, 'training max 5', 'consensus-max5-' + str(args.r) + '.pkl')
    train_save_clf(tree.DecisionTreeClassifier(max_depth=10), X, y, 'training max 10', 'consensus-max10-' + str(args.r) + '.pkl')
    train_save_clf(tree.DecisionTreeClassifier(), X, y, 'training full size', 'consensus-' + str(args.r) + '.pkl')