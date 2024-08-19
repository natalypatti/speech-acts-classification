import argparse
import os
import sys
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel

from scripts.classes.model import *
from scripts.trainer import Trainer, PseudoLabelsTrainer
from scripts.utils import *
from scripts.prepare_data import *
from torch.utils.data import DataLoader, ConcatDataset

import warnings
warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_results",
        default='src/data/outputs/results/results.csv',
        help="Path to save model results",
    )
    parser.add_argument(
        "--path_to_models",
        default='src/data/outputs/models/',
        help="Path to save model models",
    )
    parser.add_argument(
        "--do_train",
        default=True,
        help="Wether to do the training",
    )
    parser.add_argument(
        "--do_validation",
        default=True,
        help="Wether to do the evaluation",
    )
    parser.add_argument(
        "--do_evaluation",
        default=True,
        help="Wether to do the evaluation (after training)",
    )
    parser.add_argument(
        "--train_path",
        default=r"src/data/data/final_train_porttinari.csv",
        help="Path to do the training",
    )
    parser.add_argument(
        "--test_path",
        default=r"src/data/data/final_test_porttinari.csv",
        help="Path to do the test",
    )
    parser.add_argument(
        "--val_path",
        default=r"src/data/data/final_val_porttinari.csv",
        help="Path to do the val",
    )
    parser.add_argument(
        "--unlabeled_path",
        default=r"src/data/data/porttinari_unlabeled_preprocess.csv",
        help="Path to do the unlabeled sample",
    )
    parser.add_argument(
        "--labels_path",
        default=r"src/data/data/porttinari_labels.txt",
        help="Path to labels",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--resample_perc",
        default=0.1,
        type=float,
        help="Percentage of inform train examples to keep",
    )
    parser.add_argument(
        "--resample_perc_trans",
        default=0.5,
        help="Percentage of inform unlabeled train examples to keep",
    )
    parser.add_argument(
        "--use_pos",
        default=False,
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        help="Add pos to bert",
    )
    parser.add_argument(
        "--use_context",
        default=True,
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        help="Add context to bert",
    )
    parser.add_argument(
        "--use_trans",
        default=False,
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        help="Add transductive to bert",
    )
    parser.add_argument(
        "--need_to_train_trans",
        default=False,
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        help="if the pseudolabels is already generated its not necessary to train",
    )
    parser.add_argument(
        "--trans_examples_to_use",
        default=0.5,
        help="perc trans examples to use",
    )  
    parser.add_argument(
        "--ignore_some_pseudolabels",
        default=True,
        help="ignore pseudolabels 5 and 10",
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
        "--lr_rate",
        default=5e-5,
        type=float,
        help="Bert finetuning lr rate",
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
        "--use_weights",
        default=False,
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        help="Use class weights during finetuning",
    )    
    parser.add_argument(
        "--join_train_val",
        default=False,
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        help="join val and train as train dataset",
    )           

    args = parser.parse_args()
    
    model_name = str(args.pretrained_model.replace("/", "-")) + '_use_weight_' + str(args.use_weights) + '_lr_' + str(args.lr_rate) + '_resample_perc_' + str(args.resample_perc)  +'_pos_' + str(args.use_pos) + '_context_' + str(args.use_context) + '_trans_' + str(args.use_trans) + '_final'
    logger.info("Preparing do train {} model".format(model_name)) 

    set_seed(10)
    
    logger.info("Prepare data")
    train_dataloader, train_dataset = prepare_data(args.train_path, args.train_path, 'random', args.batch_size, args.pretrained_model, use_pos=args.use_pos, use_context=args.use_context, resample_perc=args.resample_perc)
    #val_dataloader = prepare_data(args.val_path, args.train_path, 'sequential', args.batch_size, args.pretrained_model, use_pos=args.use_pos, use_context=args.use_context)
    test_dataloader, _ = prepare_data(args.test_path, args.train_path, 'sequential', args.batch_size, args.pretrained_model, use_pos=args.use_pos, use_context=args.use_context)

    if args.join_train_val:
        val_dataloader, val_dataset = prepare_data(args.val_path, args.train_path, 'random', args.batch_size, args.pretrained_model, use_pos=args.use_pos, use_context=args.use_context, resample_perc=args.resample_perc)
        combined_dataset = ConcatDataset([train_dataset, val_dataset])
        train_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        val_dataloader, _ = prepare_data(args.val_path, args.train_path, 'sequential', args.batch_size, args.pretrained_model, use_pos=args.use_pos, use_context=args.use_context)
    
    logger.info("Load pretrained model: {}".format(args.pretrained_model))
    pretrained_model = BertModel.from_pretrained(args.pretrained_model)
    pseudolabels_dataloader = None
    if args.use_trans:
        if args.need_to_train_trans:        
            logger.info("Preparing transductive pseudolabels")
            aux_model = PseudoLabelsClassifier(pretrained_model, get_n_labels(args.train_path))
            aux_trainer = PseudoLabelsTrainer(aux_model, "aux", 1e-5, args.epochs, len(train_dataloader.dataset), 0.05, args.batch_size, args.use_weights, args.path_to_results, args.path_to_models, args.train_path)
            aux_model = aux_trainer.train(train_dataloader, val_dataloader,  args.do_validation)
            unlabeled_dataloader, _ = prepare_data(args.unlabeled_path, args.train_path, 'random', args.batch_size, args.pretrained_model, 'pseudolabels', use_pos=False, use_context=False)
            aux_trainer.model = aux_model
            pseudolabels = aux_trainer.bert_predict_pseudolabels(unlabeled_dataloader)
            save_add_pseudo_labels(args.unlabeled_path, pseudolabels)
            del aux_model
            del aux_trainer
            del unlabeled_dataloader
        pseudolabels_dataloader, _ = prepare_data(args.unlabeled_path.replace('.csv', '_with_pseudolabels.csv'), args.train_path, 'random', args.batch_size, args.pretrained_model, 'pseudolabels', use_pos=args.use_pos, use_context=args.use_context, resample_perc=args.resample_perc_trans, trans_examples_perc=args.trans_examples_to_use, ignore_some_pseudolabels=args.ignore_some_pseudolabels)
        


    logger.info("Train model: {}".format(args.pretrained_model))
    if args.use_pos: model = BertClassifierPOS(pretrained_model, args.in_dim, args.hidden_dim, get_n_labels(args.train_path), args.pos_count, args.freeze_bert)
    else: model = BertClassifier(pretrained_model, args.in_dim, args.hidden_dim, get_n_labels(args.train_path), args.freeze_bert)
    trainer = Trainer(model, "main", args.lr_rate, args.epochs, len(train_dataloader.dataset), args.resample_perc, args.batch_size, args.use_weights, args.train_path, args.path_to_results, args.path_to_models, args.use_pos, args.use_context, args.use_trans, args.resample_perc)
    model = trainer.train(train_dataloader, pseudolabels_dataloader, val_dataloader,  test_dataloader, args.do_validation, args.do_evaluation)