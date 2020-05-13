<p align="center">
  <img src="example.png" width="350" title="t-SNE visualization of head-dependent dependency pairs belonging to selected dependencies in English and French, projected into a syntactic subspace of Multilingual BERT, as learned on English syntax trees. Colors correspond to gold UD dependency type labels." alt="t-SNE visualization of head-dependent dependency pairs belonging to selected dependencies in English and French, projected into a syntactic subspace of Multilingual BERT, as learned on English syntax trees. Colors correspond to gold UD dependency type labels.">
</p>

# multilingual-probing-visualization

This repository is a codebase for probing and visualizing multilingual language models, specifically Multilingual BERT, based on the ACL'20 paper [Finding Universal Grammatical Relations in Multilingual BERT
](https://nlp.stanford.edu/pubs/chi2020finding.pdf).  It draws heavily from the [structural-probes](https://github.com/john-hewitt/structural-probes/) codebase of Hewitt and Manning (2019).  All code is under the Apache license.


## Installation & Getting Started

1. Clone the repository.

        git clone https://github.com/ethanachi/multilingual-probing-visualization
        cd multilingual-probing-visualization

1. [Optional] Construct a virtual environment for this project. Only `python3` is supported.

        conda create --name probe-viz
        conda activate probe-viz

1. Install the required packages. This mainly means `pytorch`, `scipy`, `numpy`, `sklearn`, etc.
   Look at [pytorch.org](https://pytorch.org) for the PyTorch installation that suits you and install it; it won't be installed via `requirements.txt`.
   Everything in the repository will use a GPU if available, but if none is available, it will detect so and just use the CPU, so use the pytorch install of your choice.

        conda install --file requirements.txt
        pip install pytorch-pretrained-bert

### Demo: Generating t-SNE visualizations

A significant portion of our paper relies on t-SNE visualizations generated from head-dependency pairs.  
We provide a **demo script** that can be used to easily produce such visualizations, using a pretrained set of
probe parameters trained on either English or a concatenation of 10 languages.
Visualizations for English→French transfer and a joint multilingual space are currently available [here](http://multilingual-probing.appspot.com/), although this may move in the near future.

1. Download data:

        bash downloadExamples.sh

This downloads pretrained probe parameters into `examples/data`, as well as example data for English and French into the `examples/{en, fr}` folders, using the name convention described earlier.
To test on other languages, download the dev split `conllu` files into similarly-named directories.

2. Process data:

        bash scripts/process_demo.sh examples/

This will write raw `.txt` files and BERT hidden state data into the `examples/` folder.

3. Generate tSNE visualizations:

        python3 run_demo.py examples/

This will write an output directory with visualizations to disk---check the output logs.

4. Run the server and navigate to `localhost:8000`:

        Visualize:
            cd examples/results/2020-5-9-19-0-29-622752/tsne-2020-5-9-19-1-14-581865
            python3 -m http.server

## The experiment config file
Experiments run with this repository are specified via `yaml` files that completely describe the experiment (except the random seed.)
In this section, we go over each top-level key of the experiment config.

### Dataset:
 - `observation_fieldnames`: the fields (columns) of the conll-formatted corpus files to be used.
   Must be in the same order as the columns of the corpus. The final two fields  must be `langs` and `embeddings`.
   Each field will be accessable as an attribute of each `Observation` class (e.g., `observation.sentence`
   contains the sequence of tokens comprising the sentence.)
 - `corpus`: The location of the train, dev, and test conll-formatted multilingual corpora files. Each of `train_path`,
   `dev_path`, `test_path` will be taken as relative to the `root` field.
 - `embeddings`: The location of the train, dev, and test pre-computed multilingual embedding files (ignored if not applicable.
 Each of `train_path`, `dev_path`, `test_path` will be taken as relative to the `root` field.
        - `type` is ignored.
 - `keys`: A list of languages to be used for each split (train, dev, and test).
 - `batch_size`: The number of observations to put into each batch for training the probe. 20 or so should be fine.

```
dataset:
  observation_fieldnames:
     - index
     - sentence
     - lemma_sentence
     - upos_sentence
     - xpos_sentence
     - morph
     - head_indices
     - governance_relations
     - secondary_relations
     - extra_info
     - langs
     - embeddings
  corpus:
    root: /u/scr/ethanchi/langs
    train_path: train.conllu
    dev_path: dev.conllu
    test_path: test.conllu
  embeddings:
    type: token
    root: /u/scr/ethanchi/hdf5
    train_path: train-multilingual.hdf5
    dev_path: dev-multilingual.hdf5
    test_path: test-multilingual.hdf5
  keys:
    train: ['fr']
    dev: ['en']
    test: ['en']
  batch_size: 20
```

### Model
 - `hidden_dim`: The dimensionality of the representations to be probed.
    The probe parameters constructed will be of shape (hidden_dim, maximum_rank)
 - `model_type`: One of `ELMo-disk`, `BERT-disk`, `ELMo-decay`, `ELMo-random-projection` as of now.
   Used to help determine which `Dataset` class should be constructed, as well as which model will construct the representations for the probe.
   The `Decay0` and `Proj0` baselines in the paper are from `ELMo-decay` and `ELMo-random-projection`, respectively.
   In the future, will be used to specify other PyTorch models.
 - `use_disk`: Set to `True` to assume that pre-computed embeddings should be stored with each `Observation`; Set to `False` to use the words in some downstream model (this is not supported yet...)
 - `model_layer`: The index of the hidden layer to be used by the probe. For example, `ELMo` models can use layers `0,1,2`; BERT-base models have layers `0` through `11`; BERT-large `0` through `23`.
 - `tokenizer`: If a model will be used to construct representations on the fly (as opposed to using embeddings saved to disk) then a tokenizer will be needed. The `type` string will specify the kind of tokenizer used.
 The `vocab_path` is the absolute path to a vocabulary file to be used by the tokenizer.

```
model:
  hidden_dim: 768 # BERT hidden dim
  model_type: BERT-disk
  use_disk: True
  model_layer: 6 # BERT-multilingual: (0,...,11)
  multilingual: True
```

### Probe, probe-training
 - `task_signature`: Specifies the function signature of the task. Supports `word_pair` for parse distance tasks, `word` for single-word tasks, and `word_label` for classification tasks (e.g. semantic roles).  Our paper uses only the `word_pair` setting.
 - `task_name`: A unique name for each task supported by the repository. Right now, this includes `parse-depth`, `parse-distance`, and `semantic-roles`.
 - `maximum_rank`: Specifies the dimensionality of the space to be projected into, if `psd_parameters=True`.
   The projection matrix is of shape (hidden_dim, maximum_rank).
   The rank of the subspace is upper-bounded by this value.
   If `psd_parameters=False`, then this is ignored.
 - `psd_parameters`: though not reported in the paper, the `parse_distance` and `parse_depth` tasks can be accomplished with a non-PSD matrix inside the quadratic form.
   All experiments for the paper were run with `psd_parameters=True`, but setting `psd_parameters=False` will simply construct a square parameter matrix. See the docstring of `probe.TwoWordNonPSDProbe` and `probe.OneWordNonPSDProbe` for more info.
 - `diagonal`: Ignored.
 - `prams_path`: The path, relative to `args['reporting']['root']`, to which to save the probe parameters.
 - `epochs`: The maximum number of epochs to which to train the probe.
   (Regardless, early stopping is performed on the development loss.)
 - `loss`: A string to specify the loss class. Right now, only `L1` is available.
    The class within `loss.py` will be specified by a combination of this and the task name, since for example distances and depths have different special requirements for their loss functions.

```
probe:
  task_signature: word_pair # word, word_pair
  task_name: parse-distance
  maximum_rank: 32
  psd_parameters: True
  diagonal: False
  params_path: predictor.params
probe_training:
  epochs: 30
  loss: L1
```

### Reporting
 - `root`: The path to the directory in which a new subdirectory should be constructed for the results of this experiment.
 - `observation_paths`: The paths, relative to `root`, to which to write the observations formatted for quick reporting later on.
 - `prediction_paths`: The paths, relative to `root`, to which to write the predictions of the model.
 - `reporting_methods`: A list of strings specifying the methods to use to report and visualize results from the experiment.
    For `parse-distance`, the valid methods are:
    - `spearmanr`
    - `uuas`
    - `image_examples`
    - `write_data` (writes data to disk in an easier-to-read format)
    - `adj_acc` (reports UUAS for prenominal and postnominal adjectives)
    - `tsne` (generates a t-SNE visualization, see the next section)
    - `pca` (generates a PCA visualization)
    - `unproj_tsne` (generates a t-SNE visualization, but using PCA for dimensionality reduction rather than the structural probe)
    - `visualize_tsne` (copies supporting HTML files to disk for easy visualization)
    When reporting `uuas`, some `tikz-dependency` examples are written to disk as well.
    Note that `image_examples` will be ignored for the test set.
```
reporting:
  root: example/results
  observation_paths:
    train_path: train.observations
    dev_path: dev.observations
    test_path: test.observations
  prediction_paths:
    train_path: train.predictions
    dev_path: dev.predictions
    test_path: test.predictions
  reporting_methods:
    - spearmanr
      #- image_examples
    - uuas
```



## Experiments on new datasets or models
Generally speaking, the following steps are necessary to run arbitrary experiments:

1. For each language that you'd like to investigate:
  1. Have a `conllu` file for the train, dev, and test splits of your dataset. These should each go in a folder named with the language code (e.g. `path_to_conllus/en/train.conllu`).

  2. Convert each `conllu` file to plain text by running:

            python3 scripts/convert_conll_to_raw.py path_to_conllus/en/train.conllu path_to_conllus/en/train.txt

     Repeat this for each split (train, dev, test) as appropriate.

  3. Write contextual word representations to disk for each of the train, dev, and test split in `hdf5` format.   The key to each `hdf5` dataset object should be `{lang}-{index}`, where `{lang}` is the language code of the sentence's language, and `{index}` is the index of the sentence in its specific `conllu` file. That is, your dataset file should look a bit like `{'en-0': <np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>, 'en-1':<np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>...}`, etc. Note here that `SEQLEN` for each sentence must be the number of tokens in the sentence as specified by the `conllx` file. To do this for Multilingual BERT, run the following script:

            python3 scripts/convert_raw_to_bert.py path_to_conllus/en/train.txt path_to_hdf5/train_multilingual.hdf5 multilingual lang

  where `lang` is the language code (e.g. `en`).  Note that all splits for a specific language should share the same `hdf5` embeddings file.

2. Edit a `config` file from `example/config` to match the paths to your data, as well as the hidden dimension and labels for the columns in the `conllx` file. For more information, please consult the experiment config section of this README.

3. Run an experiment with `python3 probing/run_experiment.py`.

## Replicating Results for the ACL'20 Paper

Here are the steps to replicate the results for our ACL'20 paper:

1. Download the train/dev/test splits for the following datasets:
  - UD_Arabic-PADT
  - UD_Chinese-GSD
  - UD_Czech-PDT
  - UD_English-EWT
  - UD_Finnish-TDT
  - UD_French-GSD
  - UD_German-GSD
  - UD_Indonesian-GSD
  - UD_Latvian-LVTB
  - UD_Persian-Seraji
  - UD_Spanish-AnCora

2. move the datasets to folders labeled with language codes in `DATAPATH`, i.e.

            DATAPATH/en/{train, dev, test}.conllu
            DATAPATH/fr/{train, dev, test}.conllu
            ...

3. remove any sentences from the train sets larger than 512 tokens (the maximum sentence length for Multilingual BERT), that is:

- `ar`: annahar.20021130.0085:p18u1
- `fi`: j016.2
- `fr`: fr-ud-train_06464

4. Convert the conllx files to sentence-per-line whitespace-tokenized files, using `scripts/convert_conll_to_raw.py`.

5. Download the random baseline:  `bash download_random_baseline.sh`.  This will download a `.tar` file with the parameters for mBertRandom, a baseline with randomly-initialized parameters.  Change the path in `scripts/convert_raw_to_bert.py` to match your download path.

4. Use `scripts/convert_raw_to_bert.py` to take the sentence-per-line whitespace-tokenized files and write BERT vectors to disk in hdf5 format.

5. Replace the data paths (and choose a results path) in the yaml configs in `acl2020/*/*` with the paths that point to your conllx and .hdf5 files as constructed in the above steps. These 270 experiment files specify the configuration of all the experiments that end up in the paper.


## Citation

If you use this repository, please cite:

    @inproceedings{chi2020finding,
      title={Finding Universal Grammatical Relations in Multilingual BERT},
      author={Chi, Ethan A and Hewitt, John and Manning, Christopher D},
      booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      year={2020}
    }
