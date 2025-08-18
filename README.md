# 16S-Mutation-Classifiers
Methods for identifying random mutations in 16S sequences.

### Usage

1. Download the most recent version of SILVA SSU Ref NR99 from [the SILVA website](https://www.arb-silva.de/no_cache/download/archive/current/). Additionally, download the most recent [SSU Ref NR99 full metadata](https://www.arb-silva.de/no_cache/download/archive/current/Exports/full_metadata/). Decompress each file and store in the same directory as the python scripts. WABI 2025 results used SILVA version 138.2.
2. Run `python filter_silva_seqs.py`. This will produce an additional file slightlly smaller than SSU Ref NR99.
3. Run `python train.py` to train each individual feature classifier on the filtered dataset.
4. Run `python train_consensus.py` to train the consensus classifier on the filtered dataset. It is currently mandatory to run `train.py` first.
5. For each evaluation dataset, run `python eval.py -f /path/to/dataset.fa` and `python eval_consensus.py -f /path/to/dataset.fa` to print a confusion matrix for each classifier.

Jupyter notebooks containing code for computing our unsuccessful features are included, but these features are not included in our train or eval scripts.
