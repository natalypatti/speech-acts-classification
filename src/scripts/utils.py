import os
import torch
import random
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from datetime import datetime
import itertools

def set_seed(seed_value=100):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def get_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

  else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")

  return device

def get_train_weights(train_path):
  train = pd.read_csv(train_path)
  weights = class_weight.compute_class_weight(class_weight ='balanced',
                                             classes =   np.unique(train['label']),
                                             y =    train['label'])
  weights = torch.Tensor(weights).to(get_device())
  return weights

def generate_context_win(df):
  all_context_win, all_context_tokens, all_context_pos, all_context_pos_ids, all_context_labels, all_context_pos_ids, item_label = [], [], [], [], [], [], []
  for idx, row in df.iterrows():
    context_win_text, context_win_tokens, context_win_pos, context_win_pos_ids, context_win_label = [], [], [], [], []
    if idx - 1 >= 0:
      if df.iloc[idx - 1]['dialogue id'] == row['dialogue id']:
        context_win_text.append(df.iloc[idx - 1]['sentences'])
        context_win_pos.append(df.iloc[idx - 1]['pos'])
        context_win_label.append(df.iloc[idx - 1]['label'])
        context_win_pos_ids.append(df.iloc[idx - 1]['pos_id'])
        context_win_tokens.append(df.iloc[idx - 1]['tokens'])
      else:
        context_win_text.append(['-'])
        context_win_pos.append(['-'])
        context_win_label.append(['-'])
        context_win_pos_ids.append(['-'])
        context_win_tokens.append(['-'])
    else:
      context_win_text.append(['-'])
      context_win_pos.append(['-'])
      context_win_label.append(['-'])
      context_win_pos_ids.append(['-'])
      context_win_tokens.append(['-'])

    context_win_text.append(row['sentences'])
    context_win_pos.append(row['pos'])
    context_win_label.append(row['label'])
    context_win_pos_ids.append(row['pos_id'])
    context_win_tokens.append(row['tokens'])
    item_label.append(row['label'])

    if idx + 1 < len(df):
      if df.iloc[idx + 1 ]['dialogue id'] == row['dialogue id']:
        context_win_text.append(df.iloc[idx + 1]['sentences'])
        context_win_pos.append(df.iloc[idx + 1]['pos'])
        context_win_label.append(df.iloc[idx + 1]['label'])
        context_win_pos_ids.append(df.iloc[idx + 1]['pos_id'])
        context_win_tokens.append(df.iloc[idx + 1]['tokens'])
      else:
        context_win_text.append(['-'])
        context_win_pos.append(['-'])
        context_win_label.append(['-'])
        context_win_pos_ids.append(['-'])
        context_win_tokens.append(['-'])
    else:
      context_win_text.append(['-'])
      context_win_pos.append(['-'])
      context_win_label.append(['-'])
      context_win_pos_ids.append(['-'])
      context_win_tokens.append(['-'])
    all_context_win.append(context_win_text)
    all_context_tokens.append(context_win_tokens)
    all_context_pos.append(context_win_pos)
    all_context_pos_ids.append(context_win_pos_ids)
    all_context_labels.append(context_win_label)

  assert len(all_context_win) == len(all_context_tokens) == len(all_context_pos) == len(all_context_pos_ids) == len(all_context_labels) == len(item_label)

  return all_context_win, all_context_tokens, all_context_pos, all_context_pos_ids, all_context_labels, item_label

def get_cat_code(data_path):
  data = pd.read_csv(data_path)
  data['função'] = data['função'].astype(str)
  data['função'] = data['função'].str.strip()
  data['função'] = pd.Categorical(data['função'])
  #data['label'] = data['função'].cat.codes
  dict_map = dict( enumerate(data['função'].cat.categories ) )
  inv_map = {v: k for k, v in dict_map.items()}
  return dict_map, inv_map

def save_results(new_results, path_results, model_name, dataset, use_pos, use_context, use_trans, use_weights, batch_size, epoch, lr_rate, resample_perc, n_classes, fold=None):
  timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  for val, col in zip([model_name, dataset, use_pos, use_context, use_trans, use_weights, 
                              batch_size, epoch, lr_rate, resample_perc, 
                              n_classes, fold, timestamp_now], ["model_name", "dataset", "use_pos", "use_context", "use_trans", "use_weights", 
                              "batch_size", "epoch", "lr_rate", "resample_perc", 
                              "n_classes", "fold", "timestamp_now"]):
    new_results[col] = val
  if os.path.exists(path_results):
    results = pd.read_csv(path_results, index_col="Unnamed: 0")
    new_results = pd.concat([results, new_results])
  
  new_results.to_csv(path_results, index=True)
      
def combine_dataloaders(loader1, loader2):
    iter_loader1 = iter(loader1)
    iter_loader2 = iter(loader2)

    for batch1, batch2 in itertools.zip_longest(iter_loader1, iter_loader2, fillvalue=None):
        if batch1 is None:
            yield batch2
        elif batch2 is None:
            yield batch1
        else:
            combined_batch = torch.cat((batch1, batch2), dim=0)
            yield combined_batch

