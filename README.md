# 16S-Mutation-Classifiers
This repository contains methods for identifying random mutations in 16S sequences. Our implementation uses Python version 3.13.5. To install the required modules, run `pip install -r requirements.txt`.

### Test dataset usage
Run the following commands sequentially. You may specify any desired value of `r`, but it must be the same in each command.
```
python train.py -r 0.05 -f test/train.fa
python eval.py -r 0.05 -f test/test.fa
```
The output of `train.py` indicates the time spent computing each feature and training each classifier. The output of `eval.py` indicates the performance of each classifier in the following format: `TN, FP, FN, TP, sensitivity, specificity`. Consensus classifiers are listed in ascending order of complexity (i.e., maximum depth restriction).

### Standard usage

1. Download the most recent version of SILVA SSU Ref NR 99 from [the SILVA Archive](https://www.arb-silva.de/archive/current/Exports). Additionally, download the most recent [SSU Ref NR 99 full metadata](https://www.arb-silva.de/archive/current/Exports/full_metadata). Decompress each file and store in the same directory as the python scripts. Our 2025 results used SILVA version 138.2.
2. Run `python filter_silva_seqs.py`. This will produce an additional file slightlly smaller than SSU Ref NR 99.
3. Run `python train.py` to train each classifier on the filtered dataset. Specify mutation rates with e.g. `-r 0.05`.
4. For each evaluation dataset, run `python eval.py -f /path/to/dataset.fa` to print a confusion matrix and sensitivity and specificity for each classifier. Ensure that you have run `train.py -r X` before running `eval.py -r X`. Include `--save` to generate `.csv` files containing feature values in ground truth and mutated sequences.

We also include our candidate feature evaluation experiments in `feature_selection.ipynb` and our full analysis notebook `analysis.ipynb`. The latter is intended to be used after running `python eval.py -r {0.1, 0.05, 0.1} --save` on SSU Ref. SSU Ref is available on the same webpage as SSU Ref NR 99.
