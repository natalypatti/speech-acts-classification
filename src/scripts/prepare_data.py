from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast
import pandas as pd
import ast
from scripts.classes.dataset import MyDataset, MyDatasetContext
from scripts.utils import get_cat_code

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)

def get_n_labels(labels_path):
    df = pd.read_csv(labels_path)
    return len(df['label'].unique())

def save_add_pseudo_labels(data_path, pseudolabels):
    df = pd.read_csv(data_path)
    df['pseudolabels'] = pseudolabels
    df.to_csv(data_path.replace('.csv', '_with_pseudolabels_cross.csv'), index=False)

def generate_context_win(df, label_col="label"):
    logger.info("Preparing context")
    all_context_win = []
    all_context_tokens = []
    all_context_pos = []
    all_context_pos_ids = []
    all_context_labels = []
    item_label = []

    for idx, row in df.iterrows():
        context_win_text = []
        context_win_tokens = []
        context_win_pos = []
        context_win_pos_ids = []
        context_win_label = []

        def get_context(index, row, default):
            if 0 <= index < len(df) and df.iloc[index]['dialogue_id'] == row['dialogue_id']:
                context_row = df.iloc[index]
                return (
                    context_row['sentences'],
                    context_row['tokens'],
                    context_row['pos'],
                    context_row['pos_id'],
                    context_row[label_col]
                )
            return default

        default_value = (['-'], ['-'], ['-'], ['-'], ['-'])

        prev_context = get_context(idx - 1, row, default_value)
        curr_context = (row['sentences'], row['tokens'], row['pos'], row['pos_id'], row[label_col])
        next_context = get_context(idx + 1, row, default_value)

        context_win_text.extend([prev_context[0], curr_context[0], next_context[0]])
        context_win_tokens.extend([prev_context[1], curr_context[1], next_context[1]])
        context_win_pos.extend([prev_context[2], curr_context[2], next_context[2]])
        context_win_pos_ids.extend([prev_context[3], curr_context[3], next_context[3]])
        context_win_label.extend([prev_context[4], curr_context[4], next_context[4]])

        all_context_win.append(context_win_text)
        all_context_tokens.append(context_win_tokens)
        all_context_pos.append(context_win_pos)
        all_context_pos_ids.append(context_win_pos_ids)
        all_context_labels.append(context_win_label)
        item_label.append(row[label_col])

    assert len(all_context_win) == len(all_context_tokens) == len(all_context_pos) == len(all_context_pos_ids) == len(all_context_labels) == len(item_label)

    return all_context_win, all_context_tokens, all_context_pos, all_context_pos_ids, all_context_labels, item_label

def prepare_data(data_path, labels_path, sampler_type, batch_size, tokenizer_name, label_column='label', use_pos=True, use_context=True, resample_perc=None, trans_examples_perc=None, ignore_some_pseudolabels=False):
    logger.info('Preparing data: {}'.format(data_path))
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    data = pd.read_csv(data_path)
    data['tokens'] = data.tokens.apply(lambda x: ast.literal_eval(str(x)))
    pos = None
    if use_pos:
        data['pos_id'] = data.pos_id.apply(lambda x: ast.literal_eval(str(x)))
        pos = list(data['pos_id'])

    if use_context:
        all_context_win, all_context_tokens, all_context_pos, all_context_pos_ids, all_context_labels, item_label = generate_context_win(data, label_column)
        data = pd.DataFrame([all_context_win, all_context_tokens, all_context_pos, all_context_pos_ids, all_context_labels, item_label]).T
        data.rename(columns={0: 'sentences', 1: 'tokens', 2: 'pos', 3: 'pos_id', 4: 'all_labels', 5:label_column}, inplace=True)
        if use_pos: pos = list(data['pos_id'])

    if trans_examples_perc:
        data = data.sample(frac=trans_examples_perc, random_state=1)
        data

    if resample_perc:
        _, class_code_map = get_cat_code(labels_path)
        logger.info('before resample {}, {}'.format(len(data), data[label_column].value_counts()))
        other_classes_df = data[data[label_column]!=class_code_map['inform']]
        inform_df= data[data[label_column]==class_code_map['inform']]
        inform_df = inform_df.sample(frac=resample_perc, random_state=1)
        data = pd.concat([inform_df, other_classes_df])
        if ignore_some_pseudolabels: data = data[(data[label_column]!=class_code_map['inform']) & (data[label_column]!=class_code_map['question'])]
        data.reset_index(inplace=True, drop=True)
        if use_pos: pos = list(data['pos_id'])

    if use_context: Dataset = MyDatasetContext
    else: Dataset = MyDataset
    
    if label_column == "label":
        dataset = Dataset(texts=list(data['sentences']), tokenizer=tokenizer, tokens=list(data['tokens']), pos_tags=pos, labels=list(data[label_column]), n_labels=get_n_labels(labels_path))
    else:
        pseudo = None
        if label_column in data.columns: pseudo = list(data[label_column])
        dataset = Dataset(texts=list(data['sentences']), tokenizer=tokenizer, tokens=list(data['tokens']), pos_tags=pos, pseudolabels=pseudo, n_labels=get_n_labels(labels_path))
  
    if sampler_type=='random': sampler = RandomSampler(data)
    elif sampler_type=='sequential': sampler = SequentialSampler(data)

    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    logger.info('Prepared sample: {}, {}'.format(len(data), data[label_column].value_counts()))

    return data_loader, dataset

    