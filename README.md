# 16S-Mutation-Classifiers
Methods for identifying random mutations in 16S sequences.

### Standard usage

1. Download the most recent version of SILVA SSU Ref NR99 from [the SILVA website](https://www.arb-silva.de/no_cache/download/archive/current/). Additionally, download the most recent [SSU Ref NR99 full metadata](https://www.arb-silva.de/no_cache/download/archive/current/Exports/full_metadata/). Decompress each file and store in the same directory as the python scripts. Our 2025 results used SILVA version 138.2.
2. Run `python filter_silva_seqs.py`. This will produce an additional file slightlly smaller than SSU Ref NR99.
3. Run `python train.py` to train each classifier on the filtered dataset. Specify mutation rates with e.g. `-r 0.05`.
4. For each evaluation dataset, run `python eval.py -f /path/to/dataset.fa` to print a confusion matrix and sensitivity and specificity for each classifier. Ensure that you have run `train.py -r X` before running `eval.py -r X`. Include `--save` to generate `.csv` files containing feature values in ground truth and mutated sequences.

We also include our candidate feature evaluation experiments in `feature_selection.ipynb` and our full analysis notebook `analysis.ipynb`. The latter is intended to be used after running `python eval.py -r X --save` on SSU Ref with `X = [0.1, 0.05, 0.01]`.
