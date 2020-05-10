"""Contains classes for computing and reporting evaluation metrics."""

from collections import defaultdict
from datetime import datetime
import os
from shutil import copyfile

from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np
import json
import sklearn.metrics
import torch
import h5py

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class Reporter:
  """Base class for reporting.

  Attributes:
    test_reporting_constraint: Any reporting method
      (identified by a string) not in this list will not
      be reported on for the test set.
  """

  def __init__(self, args):
    raise NotImplementedError("Inherit from this class and override __init__")

  def __call__(self, prediction_batches, probe, model, dataloader, split_name):
    """
    Performs all reporting methods as specifed in the yaml experiment config dict.

    Any reporting method not in test_reporting_constraint will not
      be reported on for the test set.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataloader: A DataLoader for a data split
      split_name the string naming the data split: {train,dev,test}
    """
    self.probe = probe
    self.model = model
    for method in self.reporting_methods:
      if method in self.reporting_method_dict:
        if split_name == 'test' and method not in self.test_reporting_constraint:
          tqdm.write("Reporting method {} not in test set reporting "
              "methods (reporter.py); skipping".format(method))
          continue
        tqdm.write("Reporting {} on split {}".format(method, split_name))
        self.reporting_method_dict[method](prediction_batches
            , dataloader, split_name)
      else:
        tqdm.write('[WARNING] Reporting method not known: {}; skipping'.format(method))


  def write_data(self, prediction_batches, dataset, split_name, save=True):
    """
    Writes data to disk for easier analysis.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A DataLoader for a data split
      split_name: the string naming the data split: {train,dev,test}
      save: Whether to save to disk, or to return only.
    """
    output_path = os.path.join(self.reporting_root, 'data')
    if not os.path.exists(output_path):
      os.mkdir(output_path)

    if self.args['did_train']:
      print("Probe was trained.")
    else:
      print("Probe was not trained, omitting projections...")
    to_output = ["projections", "sentences", "idxs", "words", "relations", "pos", "pairs", "diffs", "morphs", "representations", "is_head"]
    outputs = defaultdict(list)

    i = 0
    for data_batch, label_batch, length_batch, observation_batch in dataset:
      for label, length, (observation, _), representation in zip(label_batch, length_batch, observation_batch, data_batch):
        representation = self.model(representation[:length])
        head_indices = [int(x) - 1 for x in observation.head_indices]

        if self.args['did_train']:
          proj_matrix = self.probe.proj if hasattr(self.probe, 'proj') else self.probe.linear1.weight.data.transpose(0, 1)
          projection = torch.matmul(representation, proj_matrix).detach().cpu().numpy()
          projection_heads = projection[head_indices]

        to_add = {
          "sentences": [" ".join(observation.sentence)] * int(length),
          "idxs": range(representation.shape[0]),
          "words": observation.sentence,
          "langs": observation.langs,
          "relations": observation.governance_relations,
          "pos": (observation.upos_sentence if hasattr(observation, 'upos_sentence') else observation.pos),
          "is_head": [(x == '0') for x in observation.head_indices],
        }
        if save: to_add['representations'] = representation.detach().cpu().numpy()
        else: to_add['base_diffs'] = representation.detach().cpu().numpy() - representation[head_indices].detach().cpu().numpy()
        if self.args['did_train']: to_add.update({
            "projections": projection,
            "diffs": np.array(projection) - np.array(projection_heads)
        })

        if hasattr(observation, 'morph'): to_add['morphs'] = observation.morph
        if hasattr(observation, 'pred'): to_add['pred'] = observation.pred
        if hasattr(self.probe, 'linear1'): to_add['logits'] = self.probe(representation.unsqueeze(0)).detach().cpu().numpy().squeeze(axis=0)
        for target in to_add:
          outputs[target] += list(to_add[target])
        i += 1
    if save:
      for output, values in outputs.items():
        print("Writing", output, "to disk")
        if output in ('representations', 'projections', 'logits', 'diffs'):
          hf = h5py.File(os.path.join(output_path, split_name + '-' + output + '.hdf5'), 'w')
          hf.create_dataset(output, data=np.array(values), compression="gzip")
          hf.close()
        else:
          with open(os.path.join(self.reporting_root, split_name + '-' + output + '.txt'), 'w') as fout:
            fout.write("\n".join(str(item) for item in values))
    return outputs

  def write_tsne(self, prediction_batches, dataset, split_name, num_to_write=100000):
    """
    Writes a t-SNE visualization of dependencies to disk, along with a server to visualize them.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A DataLoader for a data split
      split_name: the string naming the data split: {train,dev,test}
      num_to_write: How many dependencies to visualize (chosen uniformly per-language)
    """
    now = datetime.now()
    date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]))
    output_path = os.path.join(self.reporting_root, 'tsne' + '-' + date_suffix)
    if not os.path.exists(output_path):
      os.mkdir(output_path)
    print("Constructing new tsne reporting directory at", output_path)

    outputs = self.write_data(prediction_batches, dataset, split_name, save=False)
    if 'reporting_settings' in self.args['reporting']:
        ppl = self.args['reporting']['reporting_settings']['ppl']
        print("Setting perplexity to", ppl)
    else:
        ppl = 30

    if 'reporting_settings' in self.args['reporting'] and 'training_iterations' in self.args['reporting']['reporting_settings']:
        n_iter = self.args['reporting_settings']['n_iter']
    else:
        n_iter = 1000

    tsne = TSNE(n_components=2, verbose=10, perplexity=ppl, n_iter=n_iter)
    print("Fitting TSNE.")

    keys = ['']
    if 'keys' in self.args['dataset']:
      keys = self.args['dataset']['keys'][split_name]

    # we sample uniformly per-language
    labels = outputs['relations']
    langs_to_indices = {}
    num_needed = num_to_write // len(keys)
    lang_labels = np.array(outputs['langs'])
    for lang in keys:
      indices_to_choose = np.where(lang_labels == lang)[0]
      if num_needed > indices_to_choose.shape[0]:
        print("Warning: wanted", num_needed, "examples for tSNE, but only", indices_to_choose.shape[0], "found.")
      indices_needed = indices_to_choose[np.random.choice(indices_to_choose.shape[0], min(num_needed, indices_to_choose.shape[0]), replace=False)]
      langs_to_indices[lang] = indices_needed
    indices = np.concatenate([langs_to_indices[lang] for lang in keys])

    to_write = ['relations', 'sentences', 'idxs', 'words', 'langs']
    cut_outputs = {(output_name[:-1] if output_name.endswith('s') else output_name): np.expand_dims(np.array(outputs[output_name])[indices], axis=1) for output_name in to_write}

    # Perform tSNE
    cut_diffs = np.array(outputs['diffs'])[indices]
    reduced = tsne.fit_transform(cut_diffs)
    print("Fitted.")
    tsv_out = np.concatenate((reduced, *cut_outputs.values()), axis=1)
    for cut_output, cut_vector in cut_outputs.items():
      with open(os.path.join(output_path, split_name + '-' + cut_output + '.txt'), 'w') as f:
        # print(cut_output, cut_vector)
        f.write('\n'.join(list(cut_vector.astype(str).squeeze())))
    hf = h5py.File(os.path.join(output_path, split_name + '.hdf5'), 'w')

    header = "x0\tx1\t" + "\t".join(cut_outputs.keys())
    print("Writing to", os.path.join(output_path, split_name + '.tsv'))

    for filename in ('index.html', 'main.js', 'extended.css'):
      copyfile(os.path.join('../visualization/', filename), os.path.join(output_path, filename))
    print("")

    print("Visualize:")
    print("\tcd", output_path)
    print("\tpython3 -m http.server")

    np.savetxt(os.path.join(output_path, split_name + '.tsv'), tsv_out, fmt="%s", header=header, comments="", delimiter="\t")

  def write_pca(self, prediction_batches, dataset, split_name, num_to_write=100000):
    """
    Writes a PCA visualization of dependencies to disk, along with a server to visualize them.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A DataLoader for a data split
      split_name: the string naming the data split: {train,dev,test}
      num_to_write: How many dependencies to visualize (chosen uniformly per-language)
    """
    output_path = os.path.join(self.reporting_root, 'tsne')
    if not os.path.exists(output_path):
      os.mkdir(output_path)
    outputs = self.write_data(prediction_batches, dataset, split_name, save=False)
    pca = PCA(n_components=2, random_state=229)
    print("Fitting PCA.")

    keys = ['']
    if 'keys' in self.args['dataset']:
      keys = self.args['dataset']['keys'][split_name]

    # we sample uniformly per-language
    labels = outputs['relations']
    langs_to_indices = {}
    num_needed = num_to_write // len(keys)
    lang_labels = np.array(outputs['langs'])
    for lang in keys:
      indices_to_choose = np.where(lang_labels == lang)[0]
      if num_needed > indices_to_choose.shape[0]:
        print("Warning: wanted", num_needed, "examples for tSNE, but only", indices_to_choose.shape[0], "found.")
      indices_needed = indices_to_choose[np.random.choice(indices_to_choose.shape[0], min(num_needed, indices_to_choose.shape[0]), replace=False)]
      langs_to_indices[lang] = indices_needed
    indices = np.concatenate([langs_to_indices[lang] for lang in keys])

    to_write = ['relations', 'sentences', 'idxs',	'words', 'langs']
    cut_outputs = {(output_name[:-1] if output_name.endswith('s') else output_name): np.expand_dims(np.array(outputs[output_name])[indices], axis=1) for output_name in to_write}

    # Perform tSNE
    cut_diffs = np.array(outputs['diffs'])[indices]
    reduced = pca.fit_transform(cut_diffs)
    print("Fitted.")
    tsv_out = np.concatenate((reduced, *cut_outputs.values()), axis=1)
    for cut_output, cut_vector in cut_outputs.items():
      with open(os.path.join(output_path, split_name + '-' + cut_output + '.txt'), 'w') as f:
        # print(cut_output, cut_vector)
        f.write('\n'.join(list(cut_vector.astype(str).squeeze())))
    hf = h5py.File(os.path.join(output_path, split_name + '.hdf5'), 'w')

    header = "x0\tx1\t" + "\t".join(cut_outputs.keys())
    print("Writing to", os.path.join(output_path, split_name + '.tsv'))
    np.savetxt(os.path.join(output_path, split_name + '.tsv'), tsv_out, fmt="%s", header=header, comments="", delimiter="\t")


  def write_unprojected_tsne(self, prediction_batches, dataset, split_name, num_to_write=100000):
    """
    Writes a t-SNE visualization of dependencies to disk, projected to 32 dimensions using PCA, not the structural probe.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A DataLoader for a data split
      split_name: the string naming the data split: {train,dev,test}
      num_to_write: How many dependencies to visualize (chosen uniformly per-language)
    """
    output_path = os.path.join(self.reporting_root, 'tsne')
    if not os.path.exists(output_path):
      os.mkdir(output_path)
    outputs = self.write_data(prediction_batches, dataset, split_name, save=False)
    pca = PCA(n_components=32, random_state=229)

    print("Fitting preliminary PCA.")

    keys = ['']
    if 'keys' in self.args['dataset']:
      keys = self.args['dataset']['keys'][split_name]

    # we sample uniformly per-language
    labels = outputs['relations']
    langs_to_indices = {}
    num_needed = num_to_write // len(keys)
    lang_labels = np.array(outputs['langs'])
    for lang in keys:
      indices_to_choose = np.where(lang_labels == lang)[0]
      if num_needed > indices_to_choose.shape[0]:
        print("Warning: wanted", num_needed, "examples for tSNE, but only", indices_to_choose.shape[0], "found.")
      indices_needed = indices_to_choose[np.random.choice(indices_to_choose.shape[0], min(num_needed, indices_to_choose.shape[0]), replace=False)]
      langs_to_indices[lang] = indices_needed
    indices = np.concatenate([langs_to_indices[lang] for lang in keys])

    to_write = ['relations', 'sentences', 'idxs',	'words', 'langs']
    cut_outputs = {(output_name[:-1] if output_name.endswith('s') else output_name): np.expand_dims(np.array(outputs[output_name])[indices], axis=1) for output_name in to_write}

    # Perform tSNE
    base_diffs = np.array(outputs['base_diffs'])[indices]
    cut_diffs = pca.fit_transform(base_diffs)
    print("Fitted PCA to 32 dimensions.")

    if 'reporting_settings' in self.args['reporting']:
        ppl = self.args['reporting']['reporting_settings']['ppl']
        print("Setting perplexity to", ppl)
    else:
        ppl = 30

    if 'reporting_settings' in self.args['reporting'] and 'training_iterations' in self.args['reporting']['reporting_settings']:
        n_iter = self.args['reporting_settings']['n_iter']
    else:
        n_iter = 1000

    tsne = TSNE(n_components=2, verbose=10, perplexity=ppl, n_iter=n_iter)
    reduced = tsne.fit_transform(cut_diffs)
    tsv_out = np.concatenate((reduced, *cut_outputs.values()), axis=1)
    for cut_output, cut_vector in cut_outputs.items():
      with open(os.path.join(output_path, split_name + '-' + cut_output + '.txt'), 'w') as f:
        # print(cut_output, cut_vector)
        f.write('\n'.join(list(cut_vector.astype(str).squeeze())))
    hf = h5py.File(os.path.join(output_path, split_name + '.hdf5'), 'w')

    header = "x0\tx1\t" + "\t".join(cut_outputs.keys())
    print("Writing to", os.path.join(output_path, split_name + '.tsv'))
    np.savetxt(os.path.join(output_path, split_name + '.tsv'), tsv_out, fmt="%s", header=header, comments="", delimiter="\t")

  def write_json(self, prediction_batches, dataset, split_name):
    """Writes observations and predictions to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    json.dump([prediction_batch.tolist() for prediction_batch in prediction_batches]
        , open(os.path.join(self.reporting_root, split_name+'.predictions'), 'w'))
    json.dump([[x[0][:-1] for x in observation_batch] for _,_,_, observation_batch in dataset],
        open(os.path.join(self.reporting_root, split_name+'.observations'), 'w'))

class WordPairReporter(Reporter):
  """Reporting class for wordpair (distance) tasks"""

  def __init__(self, args):
    self.args = args
    self.reporting_methods = args['reporting']['reporting_methods']
    self.reporting_method_dict = {
        'spearmanr': self.report_spearmanr,
        'image_examples':self.report_image_examples,
        'uuas':self.report_uuas_and_tikz,
        'write_predictions':self.write_json,
        'proj_acc': self.report_proj_nonproj_accuracy,
        'adj_acc': self.report_adj_accuracy,
        'write_data': self.write_data,
        'tsne': self.write_tsne,
        'pca': self.write_pca,
        'unproj_tsne': self.write_unprojected_tsne,
    }
    self.reporting_root = args['reporting']['root']
    self.test_reporting_constraint = {'spearmanr', 'uuas', 'root_acc'}

  def generate_square_dist_matrix(self, length):
    x = np.linspace(-(length-1)/2, (length-1)/2, num=length)
    weighting = np.flip(x).T[:, np.newaxis] + x # black magic
    weighting = np.abs(weighting)
    weighting = torch.FloatTensor(weighting)
    return weighting

  def report_spearmanr(self, prediction_batches, dataset, split_name):
    """Writes the Spearman correlations between predicted and true distances.

    For each word in each sentence, computes the Spearman correlation between
    all true distances between that word and all other words, and all
    predicted distances between that word and all other words.

    Computes the average such metric between all sentences of the same length.
    Writes these averages to disk.
    Then computes the average Spearman across sentence lengths 5 to 50;
    writes this average to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    lengths_to_spearmanrs = defaultdict(list)
    use_linear = False
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        length = int(length)

        # uncomment this for linear baseline
        if use_linear: prediction = self.generate_square_dist_matrix(length)
        else: prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()
        spearmanrs = [spearmanr(pred, gold) for pred, gold in zip(prediction, label)]
        lengths_to_spearmanrs[length].extend([x.correlation for x in spearmanrs])
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length])
        for length in lengths_to_spearmanrs}

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr' + ('_linear' if use_linear else '')), 'w') as fout:
      for length in sorted(mean_spearman_for_each_length):
        fout.write(str(length) + '\t' + str(mean_spearman_for_each_length[length]) + '\n')

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr-5_50-mean' + ('_linear' if use_linear else '')), 'w') as fout:
      mean = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])
      fout.write(str(mean) + '\n')

  def report_image_examples(self, prediction_batches, dataset, split_name):
    """Writes predicted and gold distance matrices to disk for the first 20
    elements of the developement set as images!

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    images_printed = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        length = int(length)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()
        words = observation.sentence
        fontsize = 5*( 1 + np.sqrt(len(words))/200)
        plt.clf()
        ax = sns.heatmap(label)
        ax.set_title('Gold Parse Distance')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='top')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reporting_root, split_name + '-gold'+str(images_printed)), dpi=300)

        plt.clf()
        ax = sns.heatmap(prediction)
        ax.set_title('Predicted Parse Distance (squared)')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reporting_root, split_name + '-pred'+str(images_printed)), dpi=300)
        print('Printing', str(images_printed))
        images_printed += 1
        if images_printed == 20:
          return

  def report_uuas_and_tikz(self, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    uspan_total = 0
    uspan_correct = 0
    total_sents = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(
        prediction_batches, dataset), desc='[uuas,tikz]'):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.upos_sentence
        length = int(length)
        assert length == len(observation.sentence)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()

        gold_edges = prims_matrix_to_edges(label, words, poses)

        # uncomment this for linear baseline
        # pred_edges = [(i, i+1) for i in range(len(words)-1)]
        pred_edges = prims_matrix_to_edges(prediction, words, poses)

        if split_name != 'test' and total_sents < 20:
          self.print_tikz(pred_edges, gold_edges, words, split_name)

        uspan_correct += len(set([tuple(sorted(x)) for x in gold_edges]).intersection(
          set([tuple(sorted(x)) for x in pred_edges])))
        uspan_total += len(gold_edges)
        total_sents += 1
    uuas = uspan_correct / float(uspan_total)
    with open(os.path.join(self.reporting_root, split_name + '.uuas'), 'w') as fout:
      fout.write(str(uuas) + '\n')



  def report_proj_nonproj_accuracy(self, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    uspan_total = 0
    uspan_correct = 0
    total_nonproj_deps = 0
    total_proj_deps = 0
    correct_nonproj_deps = 0
    correct_proj_deps = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(
        prediction_batches, dataset), desc='[proj]'):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.upos_sentence
        length = int(length)
        assert length == len(observation.sentence)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()

        ## calculate which are projective and which are non-projective
        head_indices = [int(x) for x in observation.head_indices]
        is_proj = np.zeros([length])
        def is_projective(dep):
            head = head_indices[dep]-1
            for i in range(min(dep, head) + 1, max(dep, head)):
                curr = head_indices[i]
                while curr != 0 and curr != head + 1:
                    curr = head_indices[curr-1]
                if head != -1 and curr == 0:
                    return False
            return True

        is_proj = [is_projective(dep) for dep in range(length)]

        gold_edges = prims_matrix_to_edges(label, words, poses)
        pred_edges = prims_matrix_to_edges(prediction, words, poses)

        pairs = 0

        out_debug_pairs = []

        for idx, head_idx in enumerate(head_indices):
            if head_idx == 0: continue
            if poses[idx] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-", "PUNCT"]: continue
            head_idx -= 1
            out_debug_pairs.append(tuple(sorted([idx, head_idx])))
            if is_projective(idx):
                total_proj_deps += 1
                pairs += 1
                correct_proj_deps += (tuple(sorted([idx, head_idx])) in pred_edges)
            else:
                total_nonproj_deps += 1
                pairs += 1
                correct_nonproj_deps += (tuple(sorted([idx, head_idx])) in pred_edges)

        out_debug_pairs = sorted(out_debug_pairs)
        assert gold_edges == out_debug_pairs
        verbose = False
        if verbose:
            for word, val in [kv for kv in zip(words, is_proj)]:
                if not val: print("**", end="")
                print(word, end="")
                if not val: print("**", end="")

    correct_deps = correct_proj_deps + correct_nonproj_deps
    total_deps = total_proj_deps + total_nonproj_deps

    with open(os.path.join(self.reporting_root, split_name+'-nonproj.info'), 'w') as fout:
        fout.write(f"Proj: {correct_proj_deps}\n{total_proj_deps}\n{correct_proj_deps/total_proj_deps}\n")
        if total_nonproj_deps != 0:
            fout.write(f"Nonproj: {correct_nonproj_deps}\n{total_nonproj_deps}\n{correct_nonproj_deps/total_nonproj_deps}\n")
        fout.write(f"Total: {correct_deps}\n{total_deps}\n{correct_deps/total_deps}\n")

    with open(os.path.join(self.reporting_root, split_name+'-nonproj.acc'), 'w') as fout:
        if total_nonproj_deps != 0:
            fout.write(f"{correct_nonproj_deps}\n{total_nonproj_deps}\n{correct_nonproj_deps/total_nonproj_deps}")

  def print_tikz(self, prediction_edges, gold_edges, words, split_name):
    ''' Turns edge sets on word (nodes) into tikz dependency LaTeX. '''
    with open(os.path.join(self.reporting_root, split_name+'.tikz'), 'a') as fout:
      string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
    \\begin{deptext}[column sep=0.05cm]
    """
      string += "\\& ".join([x.replace('$', '\$').replace('&', '+') for x in words]) + " \\\\" + '\n'
      string += "\\end{deptext}" + '\n'
      for i_index, j_index in gold_edges:
        string += '\\depedge{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      for i_index, j_index in prediction_edges:
        string += '\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      string += '\\end{dependency}\n'
      fout.write('\n\n')
      fout.write(string)


  def report_adj_accuracy(self, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    total_pre_adj = 0
    correct_pre_adj = 0
    total_post_adj = 0
    correct_post_adj = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(
        prediction_batches, dataset), desc='[mod]'):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        poses = observation.upos_sentence
        length = int(length)
        assert length == len(observation.sentence)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()

        head_indices = [int(x)-1 for x in observation.head_indices]
        is_proj = np.zeros([length])


        gold_edges = prims_matrix_to_edges(label, words, poses)
        pred_edges = prims_matrix_to_edges(prediction, words, poses)

        pairs = 0

        for idx, head_idx in enumerate(head_indices):
            if head_idx == -1: continue
            if poses[idx] == 'ADJ' and poses[head_idx] == 'NOUN':
                valid = True
                for i in range(min(idx, head_idx)+1, max(idx, head_idx)): # check that all words in between are adjectives or adverbs
                  print(words[i])
                  if poses[i] not in ('ADJ', 'ADV'):
                     valid = False
                     break
                if valid:
                  if head_idx > idx:
                    total_pre_adj += 1
                    print(correct_pre_adj, pred_edges, idx, head_idx)
                    correct_pre_adj += (tuple(sorted([idx, head_idx])) in pred_edges)
                  else:
                    total_post_adj += 1
                    correct_post_adj += (tuple(sorted([idx, head_idx])) in pred_edges)

    with open(os.path.join(self.reporting_root, split_name+'-adj.info'), 'w') as fout:
        if total_pre_adj:
          fout.write(f"Pre: ({correct_pre_adj}/{total_pre_adj})\t{correct_pre_adj/total_pre_adj}\n")
        if total_post_adj:
          fout.write(f"Post: ({correct_post_adj}/{total_post_adj})\t{correct_post_adj/total_post_adj}\n")

  def print_tikz(self, prediction_edges, gold_edges, words, split_name):
    ''' Turns edge sets on word (nodes) into tikz dependency LaTeX. '''
    with open(os.path.join(self.reporting_root, split_name+'.tikz'), 'a') as fout:
      string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
    \\begin{deptext}[column sep=0.05cm]
    """
      string += "\\& ".join([x.replace('$', '\$').replace('&', '+') for x in words]) + " \\\\" + '\n'
      string += "\\end{deptext}" + '\n'
      for i_index, j_index in gold_edges:
        string += '\\depedge{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      for i_index, j_index in prediction_edges:
        string += '\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i_index+1,j_index+1, '.')
      string += '\\end{dependency}\n'
      fout.write('\n\n')
      fout.write(string)

class WordReporter(Reporter):
  """Reporting class for single-word (depth) tasks"""

  def __init__(self, args):
    self.args = args
    self.reporting_methods = args['reporting']['reporting_methods']
    self.reporting_method_dict = {
        'spearmanr': self.report_spearmanr,
        'root_acc':self.report_root_acc,
        'write_predictions':self.write_json,
        'image_examples':self.report_image_examples,
        'label_accuracy':self.report_label_values,
        'confusion_matrix': self.report_confusion_matrix,
        'distributions': self.report_distributions,
        'confusion_examples': self.report_confusion_examples
        }
    self.reporting_root = args['reporting']['root']
    self.test_reporting_constraint = {'spearmanr', 'uuas', 'root_acc'}

  def report_spearmanr(self, prediction_batches, dataset, split_name):
    """Writes the Spearman correlations between predicted and true depths.

    For each sentence, computes the spearman correlation between predicted
    and true depths.

    Computes the average such metric between all sentences of the same length.
    Writes these averages to disk.
    Then computes the average Spearman across sentence lengths 5 to 50;
    writes this average to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    lengths_to_spearmanrs = defaultdict(list)
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.sentence
        length = int(length)
        prediction = prediction[:length]
        label = label[:length].cpu()
        sent_spearmanr = spearmanr(prediction, label)
        lengths_to_spearmanrs[length].append(sent_spearmanr.correlation)
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length])
        for length in lengths_to_spearmanrs}

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr'), 'w') as fout:
      for length in sorted(mean_spearman_for_each_length):
        fout.write(str(length) + '\t' + str(mean_spearman_for_each_length[length]) + '\n')

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr-5_50-mean'), 'w') as fout:
      mean = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])
      fout.write(str(mean) + '\n')

  def report_root_acc(self, prediction_batches, dataset, split_name):
    """Computes the root prediction accuracy and writes to disk.

    For each sentence in the corpus, the root token in the sentence
    should be the least deep. This is a simple evaluation.

    Computes the percentage of sentences for which the root token
    is the least deep according to the predicted depths; writes
    this value to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    total_sents = 0
    correct_root_predictions = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        length = int(length)
        label = list(label[:length].cpu())
        prediction = prediction.data[:length]
        words = observation.sentence
        poses = observation.xpos_sentence

        correct_root_predictions += label.index(0) == get_nopunct_argmin(prediction, words, poses)
        total_sents += 1

    root_acc = correct_root_predictions / float(total_sents)
    with open(os.path.join(self.reporting_root, split_name + '.root_acc'), 'w') as fout:
      fout.write('\t'.join([str(root_acc), str(correct_root_predictions), str(total_sents)]) + '\n')

  def report_image_examples(self, prediction_batches, dataset, split_name):
    """Writes predicted and gold depths to disk for the first 20
    elements of the developement set as images!

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    images_printed = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        plt.clf()
        length = int(length)
        prediction = prediction[:length]
        label = label[:length].cpu()
        words = observation.sentence
        fontsize = 6
        cumdist = 0
        for index, (word, gold, pred) in enumerate(zip(words, label, prediction)):
          plt.text(cumdist*3, gold*2, word, fontsize=fontsize, ha='center')
          plt.text(cumdist*3, pred*2, word, fontsize=fontsize, color='red', ha='center')
          cumdist = cumdist + (np.square(len(word)) + 1)

        plt.ylim(0,20)
        plt.xlim(0,cumdist*3.5)
        plt.title('LSTM H Encoder Dependency Parse Tree Depth Prediction', fontsize=10)
        plt.ylabel('Tree Depth', fontsize=10)
        plt.xlabel('Linear Absolute Position',fontsize=10)
        plt.tight_layout()
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.savefig(os.path.join(self.reporting_root, split_name + '-depth'+str(images_printed)), dpi=300)
        images_printed += 1
        if images_printed == 20:
          return

  def report_label_values(self, prediction_batches, dataset, split_name):
    total = 0
    correct = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(prediction_batch, label_batch, length_batch, observation_batch):
        label = label[:length].cpu().numpy()
        predictions = np.argmax(prediction[:length], axis=-1)
        correct += np.sum(predictions[label != -1] == label[label != -1])
        total += len(np.where(label != -1)[0])
    with open(os.path.join(self.reporting_root, split_name + '.label_acc'), 'w') as fout:
      fout.write(str(float(correct)/  total) + '\n')

  def report_distributions(self, prediction_batches, dataset, split_name):
    true_distribution = defaultdict(int)
    predicted_distribution = defaultdict(int)
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(prediction_batch, label_batch, length_batch, observation_batch):
        label = label[:length].cpu().numpy()
        predictions = np.argmax(prediction[:length], axis=-1)
        label, predictions = label[label != -1], predictions[label != -1]
        SEMANTIC_LABELS = ["ADV", "CAU", "DIR", "DIS", "EXT", "LOC", "MNR", "MOD", "NEG", "PNC", "PRD", "PRT", "REC", "TMP"]
        for l in label: true_distribution[SEMANTIC_LABELS[int(l)]] += 1
        for p in predictions: predicted_distribution[SEMANTIC_LABELS[int(p)]] += 1
    with open(os.path.join(self.reporting_root, split_name + '.distribution'), 'w') as fout:
      for l in SEMANTIC_LABELS:
        fout.write(f"{l}\t{true_distribution[l]}\t{predicted_distribution[l]}\n")

  def report_confusion_matrix(self, prediction_batches, dataset, split_name):
    confusion_matrix = np.zeros([14, 14])
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(prediction_batch, label_batch, length_batch, observation_batch):
        label = label[:length].cpu().numpy()
        predictions = np.argmax(prediction[:length], axis=-1)
        if np.where(label != -1)[0].shape[0]:
          confusion_matrix += sklearn.metrics.confusion_matrix(label[label != -1], predictions[label != -1], range(0, 14))

    SEMANTIC_LABELS = ["ADV", "CAU", "DIR", "DIS", "EXT", "LOC", "MNR", "MOD", "NEG", "PNC", "PRD", "PRT", "REC", "TMP"]

    ax = sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": 5})
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_xticks(np.arange(14))
    ax.set_yticks(np.arange(14))
    ax.set_xticklabels(SEMANTIC_LABELS, rotation=90, fontsize=6, ha='left')
    ax.set_yticklabels(SEMANTIC_LABELS, rotation=0, fontsize=6, va='top')
    plt.tight_layout()
    plt.savefig(os.path.join(self.reporting_root, split_name + '-confusion.png'), dpi=300)

    plt.clf()
    ax = sns.heatmap(confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis], annot=True, annot_kws={"size": 5})
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_xticks(np.arange(14))
    ax.set_yticks(np.arange(14))
    SEMANTIC_LABELS = ["ADV", "CAU", "DIR", "DIS", "EXT", "LOC", "MNR", "MOD", "NEG", "PNC", "PRD", "PRT", "REC", "TMP"]
    ax.set_xticklabels(SEMANTIC_LABELS, rotation=90, fontsize=6, ha='left')
    ax.set_yticklabels(SEMANTIC_LABELS, rotation=0, fontsize=6, va='top')
    plt.tight_layout()
    plt.savefig(os.path.join(self.reporting_root, split_name + '-confusion-norm.png'), dpi=300)

    print(confusion_matrix)

  def report_confusion_examples(self, prediction_batches, dataset, split_name):
    confusion_examples = defaultdict(list)

    SEMANTIC_LABELS = ["ADV", "CAU", "DIR", "DIS", "EXT", "LOC", "MNR", "MOD", "NEG", "PNC", "PRD", "PRT", "REC", "TMP"]
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(prediction_batch, label_batch, length_batch, observation_batch):
        label = label[:length].cpu().numpy()
        predictions = np.argmax(prediction[:length], axis=-1)[label != -1]
        label = label[label != -1]
        for true_label, predicted_label, index in zip(label, predictions, observation.index):
          index = int(index)
          confusion_examples[SEMANTIC_LABELS[int(true_label)] + '-' + SEMANTIC_LABELS[int(predicted_label)]].append((" ".join(observation.sentence), observation.sentence[index], index))
    example_path = os.path.join(self.reporting_root, split_name + '_examples')
    if not os.path.exists(example_path): os.mkdir(example_path)
    for pair in confusion_examples:
      with open(os.path.join(example_path, pair + '.examples'), 'w') as fout:
       print(confusion_examples[pair])
       fout.write('\n'.join('\t'.join([str(y) for y in x]) for x in confusion_examples[pair]))



class UnionFind:
  '''
  Naive UnionFind implementation for (slow) Prim's MST algorithm

  Used to compute minimum spanning trees for distance matrices
  '''
  def __init__(self, n):
    self.parents = list(range(n))
  def union(self, i,j):
    if self.find(i) != self.find(j):
      i_parent = self.find(i)
      self.parents[i_parent] = j
  def find(self, i):
    i_parent = i
    while True:
      if i_parent != self.parents[i_parent]:
        i_parent = self.parents[i_parent]
      else:
        break
    return i_parent


def prims_matrix_to_edges(matrix, words, poses):
  '''
  Constructs a minimum spanning tree from the pairwise weights in matrix;
  returns the edges.

  Never lets punctuation-tagged words be part of the tree.
  '''
  pairs_to_distances = {}
  uf = UnionFind(len(matrix))
  for i_index, line in enumerate(matrix):
    for j_index, dist in enumerate(line):
      if poses[i_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-", "PUNCT"]:
        continue
      if poses[j_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-", "PUNCT"]:
        continue
      pairs_to_distances[(i_index, j_index)] = dist
  edges = []
  for (i_index, j_index), distance in sorted(pairs_to_distances.items(), key = lambda x: x[1]):
    if uf.find(i_index) != uf.find(j_index):
      uf.union(i_index, j_index)
      edges.append((i_index, j_index))
  return edges

def get_nopunct_argmin(prediction, words, poses):
  '''
  Gets the argmin of predictions, but filters out all punctuation-POS-tagged words
  '''
  puncts = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
  original_argmin = np.argmin(prediction)
  for i in range(len(words)):
    argmin = np.argmin(prediction)
    if poses[argmin] not in puncts:
      return argmin
    else:
      prediction[argmin] = 9000
  return original_argmin
