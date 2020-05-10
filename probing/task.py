"""Contains classes describing linguistic tasks of interest on annotated data."""

import numpy as np
import torch
import string

class Task:
  """Abstract class representing a linguistic task mapping texts to labels."""

  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.

    Should be overriden in implementing classes.
    """
    raise NotImplementedError

class DummyTask:
  @staticmethod
  def labels(observation):
    return torch.zeros(0)

class ParseDistanceTask(Task):
  """Maps observations to dependency parse distances between words."""

  @staticmethod
  def labels(observation):
    """Computes the distances between all pairs of words; returns them as a torch tensor.

    Args:
      observation: a single Observation class for a sentence:
    Returns:
      A torch tensor of shape (sentence_length, sentence_length) of distances
      in the parse tree as specified by the observation annotation.
    """
    sentence_length = len(observation[0]) #All observation fields must be of same length
    distances = torch.zeros((sentence_length, sentence_length))
    # for i in range(len(observation.head_indices)): print(i+1, observation.head_indices[i])
    for i in range(sentence_length):
      # print(i)
      for j in range(i,sentence_length):
        # print(j)
        i_j_distance = ParseDistanceTask.distance_between_pairs(observation, i, j)
        distances[i][j] = i_j_distance
        distances[j][i] = i_j_distance
    return distances

  @staticmethod
  def distance_between_pairs(observation, i, j, head_indices=None):
    '''Computes path distance between a pair of words

    TODO: It would be (much) more efficient to compute all pairs' distances at once;
          this pair-by-pair method is an artefact of an older design, but
          was unit-tested for correctness...

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: one of the two words to compute the distance between.
      j: one of the two words to compute the distance between.
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer distance d_path(i,j)
    '''
    if i == j:
      return 0
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    i_path = [i+1]
    j_path = [j+1]
    i_head = i+1
    j_head = j+1
    n = 60
    while True and n > 0:
      n -= 1
      if not (i_head == 0 and (i_path == [i+1] or i_path[-1] == 0)):
        i_head = head_indices[i_head - 1]
        i_path.append(i_head)
        # print("Appending to i_path:", i_head)
      if not (j_head == 0 and (j_path == [j+1] or j_path[-1] == 0)):
        j_head = head_indices[j_head - 1]
        j_path.append(j_head)
        # print("Appending to j_path:", j_head)
      if i_head in j_path:
        j_path_length = j_path.index(i_head)
        i_path_length = len(i_path) - 1
        break
      elif j_head in i_path:
        i_path_length = i_path.index(j_head)
        j_path_length = len(j_path) - 1
        break
      elif i_head == j_head:
        i_path_length = len(i_path) - 1
        j_path_length = len(j_path) - 1
        break
    try:
      total_length = j_path_length + i_path_length
    except:
      print(observation)
      raise AssertionError
    return total_length

class ParseDepthTask:
  """Maps observations to a depth in the parse tree for each word"""

  @staticmethod
  def labels(observation):
    """Computes the depth of each word; returns them as a torch tensor.

    Args:
      observation: a single Observation class for a sentence:
    Returns:
      A torch tensor of shape (sentence_length,) of depths
      in the parse tree as specified by the observation annotation.
    """
    sentence_length = len(observation[0]) #All observation fields must be of same length
    depths = torch.zeros(sentence_length)
    for i in range(sentence_length):
      depths[i] = ParseDepthTask.get_ordering_index(observation, i)
    return depths

  @staticmethod
  def get_ordering_index(observation, i, head_indices=None):
    '''Computes tree depth for a single word in a sentence

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: the word in the sentence to compute the depth of
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer depth in the tree of word i
    '''
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    length = 0
    i_head = i+1
    while True:
      i_head = head_indices[i_head - 1]
      if i_head != 0:
        length += 1
      else:
        return length


class SemanticRolesTask:
  """Maps observations to their semantic roles."""

  @staticmethod
  def labels(observation):
    """Computes the depth of each word; returns them as a torch tensor.

    Args:
      observation: a single Observation class for a sentence:
    Returns:
      A torch tensor of shape (sentence_length, num_labels) of depths
      in the parse tree as specified by the observation annotation.
    """
    labels = [SemanticRolesTask.label_index(label) for label in observation.pred]
    labels = torch.Tensor(labels)
    return labels



    # batch_clean = lambda l: ([elem.replace('AM-', '') if elem.startswith('AM-') and elem.replace('AM-', '') in SEMANTIC_LABELS else '_' for elem in l])
    # possible_labels = [batch_clean(getattr(observation, key)) for key in observation._asdict() if key.startswith("apred") and getattr(observation, key) and None not in getattr(observation, key)]
    # possible_labels = zip(*possible_labels)
    # possible_labels = [[subelem for subelem in elem if subelem != '_'] for elem in possible_labels]
    # labels = [SEMANTIC_LABELS.index(elem[0]) if elem else -1 for elem in possible_labels]
    return torch.Tensor(labels)

  @staticmethod
  def label_index(label):
    if label == None: return -1
    if '|' in label:
        label = "|".join(set(label.split("|")))
    if '|' in label: return -1
    label = label.lstrip(string.digits + ":")
    SEMANTIC_LABELS = ["ADV", "CAU", "DIR", "DIS", "EXT", "LOC", "MNR", "MOD", "NEG", "PNC", "PRD", "PRT", "REC", "TMP"]
    label = label.replace('AM-', '').replace('PBArgM_', '').replace('argM-', '').upper()
    conversions = {
        "FIN": "PNC",
        "ATR": "PRD",
    }
    if label in conversions: label = conversions[label]
    return SEMANTIC_LABELS.index(label) if label in SEMANTIC_LABELS else -1
