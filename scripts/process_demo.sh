#!/bin/bash

# Processes files for tSNE visualization.
# Ethan Chi (ethanchi@stanford.edu), May 2020

if [ "$#" -ne 1 ]; then
    echo "Usage: ./process_demo.sh example_dir"
fi

for lang in $1/*; do
  if [ `basename $lang` == "data" ] || [ `basename $lang` == "results" ]; then continue; fi
  if [ -f "$lang/dev.conllu" ]; then
    python3 convert_conll_to_raw.py "$lang/dev.conllu" > "$lang/dev.txt"
  else
    echo "Error: $lang does not contain dev.conllu"
    exit 1
  fi
done

mkdir -p "$1/data"

echo "Converting raw text to BERT tokens..."
for lang in $1/*; do
  if [ `basename $lang` == "data" ] || [ `basename $lang` == "results" ]; then continue; fi
  python3 convert_raw_to_bert.py "$lang/dev.txt" "$1/data/dev.hdf5" multilingual `basename $lang`
done

mkdir -p "$1/results"

echo "Done."
