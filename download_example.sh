# Downloads example corpora and vectors for visualization.

# For demo purposes, also downloads pre-trained probes on BERT-large.

wget https://nlp.stanford.edu/data/ethanchi/multilingual-probing/en.params
wget https://nlp.stanford.edu/data/ethanchi/multilingual-probing/multiling.params
mv en.params multiling.params examples/data

wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-dev.conllu
wget https://github.com/UniversalDependencies/UD_French-GSD/raw/master/fr_gsd-ud-dev.conllu

mkdir -p examples/

mkdir -p examples/en
mv en_ewt-ud-dev.conllu examples/en/dev.conllu
mkdir -p examples/fr
mv fr_gsd-ud-dev.conllu examples/fr/dev.conllu

mkdir -p examples/results
mkdir -p exmaples/data
