import argparse
import os
import sys
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertModel

from scripts.classes.model import *
from scripts.trainer import Trainer, PseudoLabelsTrainer
from scripts.utils import *
from scripts.prepare_data import *
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)


def evaluate(args, model, train_path, val_dataloader, path_results, device="cuda"):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    all_logits = []
    all_labels = []

    map_labels, _ = get_cat_code(train_path)

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        #b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        b_input_ids = batch['input_ids'].to(device)
        b_attn_mask = batch['attention_mask'].to(device)
        pos_tags = batch['pos_tags'].to(device)
        b_labels = batch['labels'].type(torch.LongTensor)
        b_labels = b_labels.to(device)
        # Compute logits
        with torch.no_grad():
            if args.use_pos: logits = model(b_input_ids, b_attn_mask, pos_tags)
            else: logits = model(b_input_ids, b_attn_mask)
            

        all_logits.append(logits)
        l_labels_max = torch.argmax(b_labels, dim=1).flatten()
        all_labels.extend(list(l_labels_max.cpu().numpy()))


    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu()
    big_values, preds = torch.max(probs, dim=1)
    preds = list(preds.cpu().numpy())

    unique_labels = list(set(list(all_labels) + list(preds)))
    target_names = []
    for cod in list(map_labels.keys()):
        if cod in unique_labels: target_names.append(map_labels[cod])
    df_report = pd.DataFrame(classification_report(list(all_labels), preds, target_names=target_names, output_dict=True))
    conf_df = pd.DataFrame(confusion_matrix(list(all_labels), preds))
    df_report.to_csv(path_results, index=False)
    #conf_df.to_csv(path_conf, index=False)

    logger.info(classification_report(all_labels, preds, target_names=target_names))
    df_report = df_report.T

    return df_report, val_loss, val_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_results",
        default='src/data/outputs/results/preds.csv',
        help="Path to save model results",
    )
    parser.add_argument(
        "--path_to_model",
        default='src/data/outputs/models/best_model',
        help="Path to save model models",
    )
    parser.add_argument(
        "--test_path",
        default=r"src/data/data/final_test_porttinari.csv",
        help="Path to do the test",
    )
    parser.add_argument(
        "--train_path",
        default=r"src/data/data/final_train_porttinari.csv",
        help="Path to do the train",
    )

    parser.add_argument(
        "--labels_path",
        default=r"src/data/data/porttinari_labels.txt",
        help="Path to labels",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--pretrained_model",
        default=r"neuralmind/bert-large-portuguese-cased",
        help="pretrained model name",
    )
    parser.add_argument(
        "--hidden_dim",
        default=50,
        help="Add transductive to bert",
    )
    parser.add_argument(
        "--in_dim",
        default=1024,
        help="Add transductive to bert",
    )
    parser.add_argument(
        "--n_labels",
        default=7,
        help="number of labels",
    )
    parser.add_argument(
        "--pos_count",
        default=17,
        help="unique pos count",
    )
    parser.add_argument(
        "--freeze_bert",
        default=False,
        help="Freeze bert during finetuning",
    )
    parser.add_argument(
        "--use_pos",
        default=True,
        help="Pos",
    )
    parser.add_argument(
        "--use_context",
        default=True,
        help="Context",
    )                  

    args = parser.parse_args()

    set_seed(10)
    
    logger.info("Prepare data")
    test_dataloader, _ = prepare_data(args.test_path, args.train_path, 'sequential', args.batch_size, args.pretrained_model, use_pos=args.use_pos, use_context=args.use_context)

    logger.info("Load pretrained model: {}".format(args.pretrained_model))
    pretrained_model = BertModel.from_pretrained(args.pretrained_model)

    logger.info("Train model: {}".format(args.pretrained_model))
    if args.use_pos: model = BertClassifierPOS(pretrained_model, args.in_dim, args.hidden_dim, args.n_labels, args.pos_count, args.freeze_bert)
    else: model = BertClassifier(pretrained_model, args.in_dim, args.hidden_dim, args.n_labels, args.freeze_bert)
    model = torch.load(args.path_to_model)
    #model.load_state_dict(torch.load(args.path_to_model))
    model.eval()
    evaluate(args, model, args.train_path, test_dataloader, args.path_to_results)