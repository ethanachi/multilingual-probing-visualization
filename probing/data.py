"""
This module handles the reading of conllx files and hdf5 embeddings.

Specifies Dataset classes, which offer PyTorch Dataloaders for the
train/dev/test splits.
"""
import os
from collections import namedtuple, defaultdict

from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import h5py


class SimpleDataset:
  """Reads conllx files to provide PyTorch Dataloaders

  Reads the data from conllx files into namedtuple form to keep annotation
  information, and provides PyTorch dataloaders and padding/batch collation
  to provide access to train, dev, and test splits.

  Attributes:
    args: the global yaml-derived experiment config dictionary
    lines_to_skip: a list of 2-tuples [inclusive] of lines to omit, 1-indexed
  """
  def __init__(self, args, task, vocab={}):
    self.args = args
    self.batch_size = args['dataset']['batch_size']
    self.use_disk_embeddings = args['model']['use_disk']
    self.vocab = vocab
    lines_to_skip = args['dataset']['corpus']['lines_to_skip'] if 'lines_to_skip' in args['dataset']['corpus'] else []

    self.lines_to_skip = [range(x[0], x[1]+1) for x in lines_to_skip]
    self.lines_to_skip = [i for sublist in self.lines_to_skip for i in sublist] # flattens
    self.observation_class = self.get_observation_class(self.args['dataset']['observation_fieldnames'])
    self.train_obs, self.dev_obs, self.test_obs = self.read_from_disk()
    self.train_dataset = ObservationIterator(self.train_obs, task)
    self.dev_dataset = ObservationIterator(self.dev_obs, task)
    self.test_dataset = ObservationIterator(self.test_obs, task)

  def read_from_disk(self):
    '''Reads observations from conllx-formatted files

    as specified by the yaml arguments dictionary and
    optionally adds pre-constructed embeddings for them.

    Returns:
      A 3-tuple: (train, dev, test) where each element in the
      tuple is a list of Observations for that split of the dataset.
    '''
    train_keys = None
    dev_keys = None
    test_keys = None
    if 'keys' in self.args['dataset']:
      train_keys = self.args['dataset']['keys']['train']
      dev_keys = self.args['dataset']['keys']['dev']
      test_keys = self.args['dataset']['keys']['test']

    root_path = self.args['dataset']['corpus']['root']
    train_path = self.args['dataset']['corpus']['train_path']
    dev_path = self.args['dataset']['corpus']['dev_path']
    test_path = self.args['dataset']['corpus']['test_path']


    train_observations = self.load_keyed_conll_dataset(root_path, train_path, skip_lines=True, keys=train_keys) if self.args['train_probe'] else []
    dev_observations = self.load_keyed_conll_dataset(root_path, dev_path, skip_lines=False, keys=dev_keys)
    test_observations = [] # self.load_keyed_conll_dataset(root_path, test_path, skip_lines=False, keys=test_keys)

    train_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['train_path'])
    dev_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['dev_path'])
    test_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['test_path'])
    train_observations = self.optionally_add_embeddings(train_observations, train_embeddings_path, skip_lines=True, keys=train_keys) if self.args['train_probe'] else []
    dev_observations = self.optionally_add_embeddings(dev_observations, dev_embeddings_path, keys=dev_keys)
    # test_observations = self.optionally_add_embeddings(test_observations, test_embeddings_path, keys=test_keys)
    return train_observations, dev_observations, test_observations

  def get_observation_class(self, fieldnames):
    '''Returns a namedtuple class for a single observation.

    The namedtuple class is constructed to hold all language and annotation
    information for a single sentence or document.

    Args:
      fieldnames: a list of strings corresponding to the information in each
        row of the conllx file being read in. (The file should not have
        explicit column headers though.)
    Returns:
      A namedtuple class; each observation in the dataset will be an instance
      of this class.
    '''
    return namedtuple('Observation', fieldnames, defaults=(None,) * len(fieldnames))

  def generate_lines_for_sent(self, lines, skip_lines=False):
    '''Yields batches of lines describing a sentence in conllx.

    Args:
      lines: Each line of a conllx file.
    Yields:
      a list of lines describing a single sentence in conllx.
    '''
    buf = []
    obvs_idx = 0
    if skip_lines: self.obvs_to_skip = []
    for index, line in enumerate(lines):
      if line.startswith('#'):
        # if 'sent_id' in line:
          # print(line)
        continue
      if not line.strip():
        if buf:
          if skip_lines and index in self.lines_to_skip:
            self.obvs_to_skip.append(obvs_idx)
          else:
            yield buf
          buf = []
          obvs_idx += 1
        else:
          continue
      else:
        buf.append(line.strip())
    if buf:
      yield buf

  def remove_ranges(self, lines, head_index):
    import copy
    lines = copy.deepcopy(lines)
    index_mappings = {'0': '0'}
    for index in range(len(lines)):
      if index >= len(lines): break
      line = lines[index]
      if '-' in line[0]:
        l, r = [int(x) for x in line[0].split('-')]
        width = r - l + 1
        newLine = lines[index+1]         # copy all data but lemma, index from the first word in the range
        newLine[0] = str(l)              # the new index is the first index of the range
        newLine[1] = line[1]             # copy the lemma of the entire fused range
        possibleIndices = [l[head_index] for l in lines[index+1:index+1+width]]

        # we only keep head indices that aren't within the range
        toUse = [x for x in possibleIndices if not (l <= int(x) <= r)]
        toUse = list(dict.fromkeys(toUse))
        if len(toUse) == 0:
          print(lines[index])
          raise AssertionError
        newLine[head_index] = toUse[0] if len(toUse) == 1 else toUse

        lines[index] = newLine
        del lines[index+1:index+1+width]
        for i in range(l, r + 1):
          index_mappings[str(i)] = str(index + 1)
      else:
        index_mappings[lines[index][0]] = str(index + 1)
      lines[index][0] = str(index + 1)

    def toMapping(x):
      if isinstance(x, list): return [index_mappings[y] for y in x]
      return index_mappings[x]

    for i, line in enumerate(lines):
      line[head_index] = toMapping(line[head_index])
    return lines

  def load_keyed_conll_dataset(self, root_path, split_path, skip_lines=False, keys=None):
    if keys == None:
      print("No keys found...")
      return self.load_conll_dataset(os.path.join(root_path, split_path), skip_lines)
    else:
      output = []
      for key in keys:
        print("Loading", key, split_path)
        output += self.load_conll_dataset(os.path.join(root_path, key, split_path), skip_lines, key)
      # print("output:", len(output))
      return output

  def load_conll_dataset(self, filepath, skip_lines=False, key=""):
    '''Reads in a conllx file; generates Observation objects

    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
      filepath: the filesystem path to the conll dataset

    Returns:
      A list of Observations
    '''
    observations = []
    lines = (x for x in open(filepath))
    head_index = self.args['dataset']['observation_fieldnames'].index('head_indices')
    for buf in self.generate_lines_for_sent(lines, skip_lines):
      conllx_lines = []
      for line in buf:
        conllx_lines.append(line.strip().split('\t'))
      conllx_lines = [x for x in conllx_lines if '.' not in x[0]]
      conllx_lines = self.remove_ranges(conllx_lines, head_index)

      data = list(zip(*conllx_lines))

      head_indices = list(data[head_index])

      # resolve ambiguities that arise with multiwords
      for i, indices in enumerate(head_indices, 1):
        if not isinstance(indices, list): continue # nothing to be resolved
        indices = list(dict.fromkeys(indices))    # remove duplicates; can't use set because want to preserve order
        if len(indices) == 0:
          raise AssertionError
        if '0' in indices:
          head_indices[i-1] = '0'
          continue
        for idx in indices:
          if (head_indices[int(idx)-1] == str(i) or
             (isinstance(head_indices[int(idx)-1], list) and str(i) in head_indices[int(idx)-1])):
            indices.remove(idx)
        if len(indices) == 1:
          head_indices[i-1] = indices[0]
        elif len(indices) == 0:
          raise AssertionError
        else:
          # print("Remaining ambiguity found", len(indices), conllx_lines[i-1])
          head_indices[i-1] = indices[-1]
      data[head_index] = tuple(head_indices)
      for x in head_indices:
        assert(isinstance(x, str)), (data, x)

      langs = [key for x in range(len(conllx_lines))]
      embeddings = [None for x in range(len(conllx_lines))]
      observation = self.observation_class(*data, langs, embeddings)
      observations.append(observation)

    return observations

  def add_embeddings_to_observations(self, observations, embeddings):
    '''Adds pre-computed embeddings to Observations.

    Args:
      observations: A list of Observation objects composing a dataset.
      embeddings: A list of pre-computed embeddings in the same order.

    Returns:
      A list of Observations with pre-computed embedding fields.
    '''
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
      embedded_observation = self.observation_class(*(observation[:-1]), embedding)
      embedded_observations.append(embedded_observation)
    return embedded_observations

  def generate_token_embeddings_from_hdf5(self, args, observations, filepath, layer_index):
    '''Reads pre-computed embeddings from ELMo-like hdf5-formatted file.

    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, sent_length, feature_count)

    Args:
      args: the global yaml-derived experiment config dictionary.
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used. (e.g., 0, 1, 2 for ELMo0, ELMo1, ELMo2.)

    Returns:
      A list of numpy matrices; one for each observation.

    Raises:
      AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
    '''
    hf = h5py.File(filepath, 'r')
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    for index in sorted([int(x) for x in indices]):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[layer_index]
      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def integerize_observations(self, observations):
    '''Replaces strings in an Observation with integer Ids.

    The .sentence field of the Observation will have its strings
    replaced with integer Ids from self.vocab.

    Args:
      observations: A list of Observations describing a dataset

    Returns:
      A list of observations with integer-lists for sentence fields
    '''
    new_observations = []
    if self.vocab == {}:
      raise ValueError("Cannot replace words with integer ids with an empty vocabulary "
          "(and the vocabulary is in fact empty")
    for observation in observations:
      sentence = tuple([vocab[sym] for sym in observation.sentence])
      new_observations.append(self.observation_class(sentence, *observation[1:]))
    return new_observations

  def get_train_dataloader(self, shuffle=True, use_embeddings=True):
    """Returns a PyTorch dataloader over the training dataset.

    Args:
      shuffle: shuffle the order of the dataset.
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the training dataset (possibly shuffled)
    """
    return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

  def get_dev_dataloader(self, use_embeddings=True):
    """Returns a PyTorch dataloader over the development dataset.

    Args:
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the development dataset
    """
    return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def get_test_dataloader(self, use_embeddings=True):
    """Returns a PyTorch dataloader over the test dataset.

    Args:
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the test dataset
    """
    return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    """Does not add embeddings; see subclasses for implementations."""
    return observations

  def custom_pad(self, batch_observations):
    '''Pads sequences with 0 and labels with -1; used as collate_fn of DataLoader.

    Loss functions will ignore -1 labels.
    If labels are 1D, pads to the maximum sequence length.
    If labels are 2D, pads all to (maxlen,maxlen).

    Args:
      batch_observations: A list of observations composing a batch

    Return:
      A tuple of:
          input batch, padded
          label batch, padded
          lengths-of-inputs batch, padded
          Observation batch (not padded)
    '''
    if self.use_disk_embeddings:
      seqs = [x[0].embeddings.to(device=self.args['device']) for x in batch_observations]
    else:
      seqs = [torch.tensor(x[0].sentence, device=self.args['device']) for x in batch_observations]
    lengths = torch.tensor([len(x) for x in seqs], device=self.args['device'])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    label_shape = batch_observations[0][1].shape
    maxlen = int(max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_maxshape, device=self.args['device']) for x in seqs]
    for index, x in enumerate(batch_observations):
      length = x[1].shape[0]
      if len(label_shape) == 1:
        labels[index][:length] = x[1]
      elif len(label_shape) == 2:
        labels[index][:length,:length] = x[1]
      else:
        raise ValueError("Labels must be either 1D or 2D right now; got either 0D or >3D")
    labels = torch.stack(labels)
    return seqs, labels, lengths, batch_observations

class ELMoDataset(SimpleDataset):
  """Dataloader for conllx files and pre-computed ELMo embeddings.

  See SimpleDataset.
  Assumes embeddings are aligned with tokens in conllx file.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    """Adds pre-computed ELMo embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print('Loading ELMo Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_token_embeddings_from_hdf5(self.args, observations, pretrained_embeddings_path, layer_index)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations

class SubwordDataset(SimpleDataset):
  """Dataloader for conllx files and pre-computed ELMo embeddings.

  See SimpleDataset.
  Assumes we have access to the subword tokenizer.
  """

  @staticmethod
  def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    '''Aligns tokenized and untokenized sentence given subwords "##" prefixed

    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.

    Args:
      tokenized_sent: a list of strings describing a subword-tokenized sentence
      untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
      A dictionary of type {int: list(int)} mapping each untokenized sentence
      index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    # print(untokenized_sent)
    # print(tokenized_sent)
    while (untokenized_sent_index < len(untokenized_sent) and
        tokenized_sent_index < len(tokenized_sent)):
      while (tokenized_sent_index + 1 < len(tokenized_sent) and
          tokenized_sent[tokenized_sent_index + 1].startswith('##')):
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        tokenized_sent_index += 1
      mapping[untokenized_sent_index].append(tokenized_sent_index)
      untokenized_sent_index += 1
      tokenized_sent_index += 1
    # print(mapping)
    return mapping

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, subword_tokenizer=None):
    raise NotImplementedError("Instead of making a SubwordDataset, make one of the implementing classes")

class BERTDataset(SubwordDataset):
  """Dataloader for conllx files and pre-computed BERT embeddings.

  See SimpleDataset.
  Attributes:
    args: the global yaml-derived experiment config dictionary
  """

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, subword_tokenizer=None, skip_lines=False, keys=None):
    '''Reads pre-computed subword embeddings from hdf5-formatted file.

    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, subword_sent_length, feature_count)
    subword_sent_length is the length of the sequence of subword tokens
    when the subword tokenizer was given each canonical token (as given
    by the conllx file) independently and tokenized each. Thus, there
    is a single alignment between the subword-tokenized sentence
    and the conllx tokens.

    Args:
      args: the global yaml-derived experiment config dictionary.
      observations: A list of Observations composing a dataset.
      filepath: The filepath of a hdf5 file containing embeddings.
      layer_index: The index corresponding to the layer of representation
          to be used. (e.g., 0, 1, 2 for BERT0, BERT1, BERT2.)
      subword_tokenizer: (optional) a tokenizer used to map from
          conllx tokens to subword tokens.

    Returns:
      A list of numpy matrices; one for each observation.

    Raises:
      AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
      Exit: importing pytorch_pretrained_bert has failed, possibly due
          to downloading of prespecifed tokenizer problem. Not recoverable;
          exits immediately.
    '''
    if subword_tokenizer == None:
      try:
        from pytorch_pretrained_bert import BertTokenizer
        if 'multilingual' in self.args['model'] and self.args['model']['multilingual'] == True:
            subword_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            print('Using BERT-base-multilingual tokenizer to align embeddings with PTB tokens')
        elif self.args['model']['hidden_dim'] == 768:
          subword_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
          print('Using BERT-base-cased tokenizer to align embeddings with PTB tokens')
        elif self.args['model']['hidden_dim'] == 1024:
          subword_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
          print('Using BERT-large-cased tokenizer to align embeddings with PTB tokens')
        else:
          print("The heuristic used to choose BERT tokenizers has failed...")
          exit()
      except:
        print('Couldn\'t import pytorch-pretrained-bert. Exiting...')
        exit()
    hf = h5py.File(filepath, 'r')
    indices = list(hf.keys())
    single_layer_features_list = []
    joiner = ' ' if 'use_no_spaces' in self.args['model'] and self.args['model']['use_no_spaces'] == True else ' '
    offset = 0
    if keys == None: keys = ['']
    running_index = 0
    # print([observation.sentence for observation in observations])
    for key in keys:
      key_indices = [index.replace(key + '-', '') for index in indices if index.startswith(key)]
      # print("key_indices are", key_indices)
      for index in tqdm(sorted([int(x) for x in key_indices]), desc='[aligning {} embeddings]'.format(key)):
        if skip_lines and index in self.obvs_to_skip:
          offset += 1
          continue
        # print(index-offset+running_index)
        observation = observations[index-offset + running_index]
        hf_key = (key + '-' + str(index)) if key else str(index)
        feature_stack = hf[hf_key]
        single_layer_features = feature_stack[elmo_layer]
        tokenized_sent = subword_tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + joiner.join(observation.sentence) + ' [SEP]')
        untokenized_sent = observation.sentence
        untok_tok_mapping = self.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)
        # print(single_layer_features.shape[0], len(tokenized_sent))
        # print(observation.sentence, observation.morph)
        assert single_layer_features.shape[0] == len(tokenized_sent)
        single_layer_features = torch.tensor([np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])
        assert single_layer_features.shape[0] == len(observation.sentence)
        single_layer_features_list.append(single_layer_features)
      running_index += index + 1
    print("single layer features list is", len(single_layer_features_list))
    return single_layer_features_list

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path, skip_lines=False, keys=None):
    """Adds pre-computed BERT embeddings from disk to Observations."""
    layer_index = self.args['model']['model_layer']
    print('Loading BERT Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_subword_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index, skip_lines=skip_lines, keys=keys)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations


class ObservationIterator(Dataset):
  """ List Container for lists of Observations and labels for them.

  Used as the iterator for a PyTorch dataloader.
  """

  def __init__(self, observations, task):
    self.observations = observations
    self.set_labels(observations, task)

  def set_labels(self, observations, task):
    """ Constructs aand stores label for each observation.

    Args:
      observations: A list of observations describing a dataset
      task: a Task object which takes Observations and constructs labels.
    """
    self.labels = []
    for observation in tqdm(observations, desc='[computing labels]'):
      self.labels.append(task.labels(observation))

  def __len__(self):
    return len(self.observations)

  def __getitem__(self, idx):
    return self.observations[idx], self.labels[idx]
