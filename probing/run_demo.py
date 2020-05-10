"""Loads configuration yaml and runs an experiment."""
from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
from glob import glob

import data
import model
import probe
import regimen
import reporter
import task
import loss

def run_report_results(args, probe, dataset, model, loss, reporter, regimen):
  probe_params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])
  dev_dataloader = dataset.get_dev_dataloader()
  try:
    probe.load_state_dict(torch.load(probe_params_path))
    probe.eval()
    dev_predictions = regimen.predict(probe, model, dev_dataloader)
  except FileNotFoundError:
    print("No trained probe found.")
    dev_predictions = None

  reporter(dev_predictions, probe, model, dev_dataloader, 'dev')

def execute_experiment(args):
  """
  Execute an experiment as determined by the configuration
  in args.
  """
  task_class = task.ParseDistanceTask
  reporter_class = reporter.WordPairReporter
  loss_class = loss.L1DistanceLoss
  dataset_class = data.BERTDataset
  probe_class = probe.TwoWordPSDProbe
  model_class = model.DiskModel
  probe_params_path = os.path.join(args['reporting']['root'],args['probe']['params_path'])
  regimen_class = regimen.ProbeRegimen


  expt_task = task_class()
  expt_dataset = dataset_class(args, expt_task)
  expt_reporter = reporter_class(args)
  expt_probe = probe_class(args)
  expt_model = model_class(args)
  expt_regimen = regimen_class(args)
  expt_loss = loss_class(args)

  print('Reporting tSNE...')
  run_report_results(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)


def setup_new_experiment_dir(yaml_args, results_dir):
  """Constructs a directory in which results and params will be stored.

  If reuse_results_path is not None, then it is reused; no new
  directory is constrcted.

  Args:
    args: the command-line arguments:
    yaml_args: the global config dictionary loaded from yaml
    reuse_results_path: the (optional) path to reuse from a previous run.
  """
  now = datetime.now()
  date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]))
  new_root = os.path.join(results_dir, date_suffix +'/')
  tqdm.write('Constructing new results directory at {}'.format(new_root))
  yaml_args['reporting']['root'] = new_root
  os.makedirs(new_root, exist_ok=True)


if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('dir')
  argp.add_argument('--langs', default=[], action='append', help='a list of languages to generate a visualization for (leave empty for all)')
  argp.add_argument('--seed', default=0, type=int, help='sets all random seeds for (within-machine) reproducibility')

  cli_args = argp.parse_args()
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  with open('../example_config.yaml') as f:
    yaml_args = yaml.load(f)

  lang_list = cli_args.langs
  if lang_list == []:
    lang_list = [os.path.basename(x) for x in glob(os.path.join(cli_args.dir, '*'))]
    lang_list.remove('data')
    lang_list.remove('results')

  print("Processing langs:", ', '.join(lang_list))

  # Dynamically generate some of our arguments
  yaml_args['dataset']['corpus'] = {
    'root': cli_args.dir,
    'train_path': '',
    'dev_path': 'dev.conllu',
    'test_path': ''
  }

  yaml_args['dataset']['embeddings'] = {
    'root': os.path.join(cli_args.dir, 'data'),
    'train_path': '',
    'dev_path': 'dev.hdf5',
    'test_path': ''
  }

  yaml_args['dataset']['keys'] = {
    'train': [],
    'dev': lang_list,
    'test': []
  }

  results_dir = os.path.join(cli_args.dir, 'results')
  setup_new_experiment_dir(yaml_args, results_dir)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  yaml_args['train_probe'] = False
  yaml_args['did_train'] = True
  execute_experiment(yaml_args)
