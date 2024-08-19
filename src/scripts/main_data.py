import argparse
import random
import pandas as pd
import ast
import logging

logger = logging.getLogger(__name__)

def pos_to_ids(data, all_pos=None, pos2id=None):
    if not all_pos:
        all_pos = []
        for pos in data['pos']:
            all_pos.extend(pos)

    if not pos2id:
        unique_pos =list(dict.fromkeys(all_pos))
        pos2id = {}
        id2pos = {}
        value = 0

        for pos in unique_pos:
            pos2id[pos]=value
            id2pos[value]=pos
            value+=1

    pos_ids = []
    for list_pos in data['pos']:
        pos_ids_temp = []
        for pos in list_pos: pos_ids_temp.append(pos2id[pos])
        pos_ids.append(pos_ids_temp)

    data['pos_id'] = pos_ids

    return data, all_pos, pos2id

def prepare_pos(data, data_unlabeled):
    data['tokens'] = data.tokens.apply(lambda x: ast.literal_eval(str(x)))
    data['pos'] = data.pos.apply(lambda x: ast.literal_eval(str(x)))

    data, all_pos, pos2id = pos_to_ids(data)

    data_unlabeled['tokens'] = data_unlabeled.tokens.apply(lambda x: ast.literal_eval(str(x)))
    data_unlabeled['pos'] = data_unlabeled.pos.apply(lambda x: ast.literal_eval(str(x)))
    data_unlabeled, _, _ = pos_to_ids(data_unlabeled, all_pos, pos2id)
    
    return data, data_unlabeled

def prepare_label(data):
    data['função'] = data['função'].str.strip()
    logger.info(data['função'].value_counts())

    # remove minor cats
    #data = data[~data['função'].isin(['apology', 'congratulation'])]
    #data = data.groupby("função").filter(lambda x: len(x) >= 12)
    count = data['função'].value_counts()
    minor_cats = count[count < 8].index
    data['função'] = data['função'].apply(lambda x: 'other' if x in minor_cats else x)
    logger.info(data['função'].value_counts())
    data['função'] = data['função'].str.strip()
    data['função'] = pd.Categorical(data['função'])
    data['label'] = data['função'].cat.codes

    return data

def get_samples(data, train_perc, val_perc):
    """Get X_train, X_val, X_test keeping all doc sentences"""
    
    df_unique_ids = data['dialogue_id'].unique()

    train_len = round(len(df_unique_ids) * train_perc)
    test_len = len(df_unique_ids) - train_len
    val_len = round(train_len * val_perc)
    train_len = train_len - val_len

    train_docs_ids = df_unique_ids[:train_len]
    test_docs_ids = df_unique_ids[train_len:train_len+test_len]
    val_docs_ids = df_unique_ids[train_len+test_len:]

    X_train = data[data['dialogue_id'].isin(train_docs_ids)]
    X_test = data[data['dialogue_id'].isin(test_docs_ids)]
    X_val = data[data['dialogue_id'].isin(val_docs_ids)]

    X_val = X_val.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    logger.info(X_train['função'].value_counts())
    logger.info(X_val['função'].value_counts())
    logger.info(X_test['função'].value_counts())

    return X_train, X_test, X_val

def remove_some_cats(train, test, val, all_cats):
    train_cats = list(train['função'].unique())

    test = test[test['função'].isin(train_cats)]
    val = val[val['função'].isin(train_cats)]

    logger.info("Removed cats {}".format(set(all_cats)-set(train_cats)))

    return train, test, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_labeled_data",
        default='src/data/data/porttinari_labeled_final.csv',
        help="Complete labeled data",
    )

    parser.add_argument(
        "--path_unlabeled_data",
        default='src/data/data/porttinari_unlabeled.csv',
        help="Complete unlabeled data",
    )

    parser.add_argument(
        "--path_outputs",
        default='src/data/data/',
        help="Output path",
    )

    parser.add_argument(
        "--dataset_name",
        default='porttinari',
        help="Dataset name",
    )

    parser.add_argument(
        "--train_perc",
        default=0.8,
        help="Training size",
    )

    parser.add_argument(
        "--val_perc",
        default=0.2,
        help="Val perc from training size",
    )

    logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)

    args = parser.parse_args()
    df_labeled  = pd.read_csv(args.path_labeled_data)
    df_unlabeled  = pd.read_csv(args.path_unlabeled_data)

    df_labeled, df_unlabeled = prepare_pos(df_labeled, df_unlabeled)
    df_labeled = prepare_label(df_labeled)
    X_train, X_test, X_val = get_samples(df_labeled, args.train_perc, args.val_perc)

    logger.info('train: {}\ntest: {}\n val: {}\n unlabeled: {}'.format(len(X_train), len(X_test), len(X_val), len(df_unlabeled)))

    X_train, X_test, X_val = remove_some_cats(X_train, X_test, X_val, list(df_labeled['função'].unique()))

    logger.info('train: {}\ntest: {}\n val: {}\n unlabeled: {}'.format(len(X_train), len(X_test), len(X_val), len(df_unlabeled)))

    X_train.to_csv(args.path_outputs + "final_train_" + args.dataset_name + ".csv", index=False)
    X_test.to_csv(args.path_outputs + "final_test_" + args.dataset_name + ".csv", index=False)
    X_val.to_csv(args.path_outputs + "final_val_" + args.dataset_name + ".csv", index=False)
    df_unlabeled.to_csv(args.path_outputs + args.dataset_name + "_unlabeled_preprocess.csv", index=False)
