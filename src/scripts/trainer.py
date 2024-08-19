import random
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from scripts.prepare_data import get_n_labels
from scripts.utils import set_seed, get_device, get_train_weights, get_cat_code, save_results
from scripts.transductive_utis import *

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)

class Trainer():
    def __init__(self, model, model_name, lr_rate, epochs, train_size, perc, batch_size, use_weights, train_path, results_path, models_path, use_pos, use_context, use_trans, resample_perc, fold=None):
        self.model = model
        self.model_name = model_name
        self.lr_rate = lr_rate
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(),
                        lr=self.lr_rate,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )
        self.total_steps = train_size * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=self.total_steps)
        self.device = get_device()
        self.results_path = results_path
        self.models_path = models_path
        self.perc = perc
        self.batch_size = batch_size
        self.use_weights = use_weights
        self.use_pos = use_pos
        self.use_context = use_context
        self.use_trans = use_trans
        self.resample_perc = resample_perc
        self.train_path = train_path
        map_labels, _ = get_cat_code(train_path)
        self.map_labels = map_labels
        self.fold = fold

        if self.use_weights: 
            self.loss_fn = nn.CrossEntropyLoss(weight=get_train_weights(train_path))
    

    def train(self, train_dataloader, train_dataloader_unlabeled=None, val_dataloader=None,  test_dataloader=None, evaluation=False, test=False):
        """
            Train the BertClassifier model.
        """
        # Start training loop
        logger.info("Start training...\n")
        set_seed()    # Set seed for reproducibility
        best_f1 = 0
        max_pred_labels = 0
        best_model = None
        
        self.model.to(self.device)
        for epoch_i in range(self.epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            logger.info("-"*70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.model.train()

            # For each batch of training data...
            #for step, batch in enumerate(train_dataloader):
            for step, batch in enumerate(train_dataloader):
                batch_counts +=1
                # Load batch to GPU
                b_input_ids = batch['input_ids'].to(self.device)
                #print(b_input_ids)
                b_attn_mask = batch['attention_mask'].to(self.device)
                pos_tags = batch['pos_tags'].to(self.device)
                b_labels = batch['labels'].type(torch.LongTensor)
                b_labels = b_labels.to(self.device)

                #b_inputs_ids_un, b_attn_mask_un, pos_tags_un, b_labels_un = batch_un['input_ids'].to(device), batch_un['attention_mask'].to(device), batch_un['pos_tags'].to(device), batch_un['pseudolabels'].type(torch.LongTensor)
                #b_labels_un = b_labels_un.to(device)
                #b_input_ids, b_attn_mask, pos_tags, b_labels = tuple(t.to(device) for t in batch)

                # Zero out any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass. This will return logits.
                if self.use_pos: logits = self.model(b_input_ids, b_attn_mask, pos_tags)
                else: logits = self.model(b_input_ids, b_attn_mask)
                loss_label = self.loss_fn(logits, torch.argmax(b_labels, dim=1).flatten())
                #print(logits)
                if train_dataloader_unlabeled:
                    # Perform a forward pass. This will return logits.
                    pseudolabels, logits_un_pseudo = preds_un(self.model, train_dataloader_unlabeled, self.device)
                    loss_transdutive = transductive_loss(pseudolabels, logits_un_pseudo, self.device)
                    p_lambda = 2
                    loss = loss_label + p_lambda * loss_transdutive
                else: loss = loss_label

                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    logger.info(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            logger.info("-"*70)

            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                # Compute predicted probabilities on the val set
                logger.info("EVALUTALION")
                df_report, val_loss, val_accuracy = self.evaluate(self.model, val_dataloader, epoch_i, val=True)
                #df_report.rename(columns={0:'precision', 1:'recall', 2:'f1', 3:'count'}, inplace=True)
                #print(df_report)
                if best_f1 < df_report.loc['macro avg']['f1-score']:
                    logger.info('BEST MODEL F1: {} epoch {} new f1 {} previous f1 {} max_pred_labels'.format(epoch_i, df_report.loc['macro avg']['f1-score'], best_f1, max_pred_labels))
                    torch.save(self.model, self.models_path + '{}_balanced_epoch_{}_perc_{}_batch_size_{}_weights_{}'.format(self.model_name, epoch_i, str(self.perc), self.batch_size, str(self.use_weights)))
                    best_f1 =  df_report.loc['macro avg']['f1-score']
                    if test:
                        logger.info("TEST")
                        _, _, _ = self.evaluate(self.model, test_dataloader, epoch_i, test=True)                
                if max_pred_labels < len(df_report[df_report['f1-score']>0.0]):
                    logger.info('BEST MODEL MAX PRED: {} epoch {} new max_pred_labels {} previous max_pred_labels {} new f1'.format(epoch_i, len(df_report[df_report['f1-score']>0.0]), max_pred_labels, best_f1))
                    max_pred_labels =  len(df_report[df_report['f1-score']>0.0])
                    torch.save(self.model, self.models_path + '{}_balanced_epoch_{}_perc_{}_batch_size_{}_weights_{}'.format(self.model_name, epoch_i, str(self.perc), self.batch_size, str(self.use_weights)))
                    if test:
                        logger.info("TEST")
                        _, _, _ = self.evaluate(self.model, test_dataloader, epoch_i, test=True)
                logger.info(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                logger.info("-"*70)
            else:
                if test:
                    logger.info("TEST")
                    _, _, _ = self.evaluate(self.model, test_dataloader, epoch_i, test=True)
            torch.save(self.model, self.models_path + "best_model")    
            logger.info("\n")

        logger.info("Training complete!")
        return self.model

    def evaluate(self, model, val_dataloader, epoch_i, val=False, test=False):
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
        
        comp_name = None
        if val: comp_name = 'val'
        elif test: comp_name = 'test'

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            #b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            b_input_ids = batch['input_ids'].to(self.device)
            b_attn_mask = batch['attention_mask'].to(self.device)
            pos_tags = batch['pos_tags'].to(self.device)
            b_labels = batch['labels'].type(torch.LongTensor)
            b_labels = b_labels.to(self.device)
            # Compute logits
            with torch.no_grad():
                if self.use_pos: logits = self.model(b_input_ids, b_attn_mask, pos_tags)
                else: logits = self.model(b_input_ids, b_attn_mask)
                

            all_logits.append(logits)

            # Compute loss
            loss = self.loss_fn(logits, torch.argmax(b_labels, dim=1).flatten())
            #loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()
            
            l_labels_max = torch.argmax(b_labels, dim=1).flatten()
            #print(preds, l_labels_max)
            # Calculate the accuracy rate
            accuracy = (preds == l_labels_max).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
            all_labels.extend(list(l_labels_max.cpu().numpy()))

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu()
        big_values, preds = torch.max(probs, dim=1)
        preds = list(preds.cpu().numpy())

        #path_report = self.results_path + '{}_balanced_epoch_{}_batch_size_{}_{}.csv'.format(self.model_name, epoch_i, str(self.perc), self.batch_size,  comp_name)
        #path_conf = self.results_path + '{}_balanced_epoch_{}_batch_size_{}_{}_conf_matrix_{}.csv'.format(self.model_name, epoch_i, str(self.perc), self.batch_size, comp_name)
        
        unique_labels = list(set(list(all_labels) + list(preds)))
        target_names = []
        for cod in list(self.map_labels.keys()):
            if cod in unique_labels: target_names.append(self.map_labels[cod])
        df_report = pd.DataFrame(classification_report(list(all_labels), preds, target_names=target_names, output_dict=True))
        conf_df = pd.DataFrame(confusion_matrix(list(all_labels), preds))
        #df_report.to_csv(path_report, index=False)
        #conf_df.to_csv(path_conf, index=False)

        save_results(df_report.T, self.results_path, self.model_name, comp_name, self.use_pos, self.use_context, self.use_trans, self.use_weights, self.batch_size, epoch_i, self.lr_rate, self.resample_perc, get_n_labels(self.train_path), self.fold)
        logger.info(classification_report(all_labels, preds, target_names=target_names))
        df_report = df_report.T

        return df_report, val_loss, val_accuracy
    

class PseudoLabelsTrainer():
    
    def __init__(self, model, model_name, lr_rate, epochs, train_size, perc, batch_size, use_weights, results_path, models_path, train_path, fold=None):
        self.model = model
        self.model_name = model_name
        self.lr_rate = lr_rate
        self.epochs = epochs
        self.device = get_device()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(),
                        lr=self.lr_rate,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )
        self.total_steps = train_size * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=self.total_steps)
        self.device = get_device()
        self.results_path = results_path
        self.models_path = models_path
        self.perc = perc
        self.batch_size = batch_size
        self.use_weights = use_weights
        self.train_path = train_path
        self.fold = fold
    
    
    def train(self, train_dataloader, val_dataloader, evaluation):
        set_seed()
        best_f1 = 0
        max_pred_labels = 0
        best_model = None
        self.model.to(self.device)

        logger.info("Training transdutive model") 
        for epoch_i in range(self.epochs+1):
            t0_epoch =  time.time()
            self.model.train()
            train_loss, train_acc = [], []
            for batch in train_dataloader:
                b_input_ids = batch['input_ids'].to(self.device)
                b_attn_mask = batch['attention_mask'].to(self.device)
                pos_tags = batch['pos_tags'].to(self.device)
                b_labels = batch['labels'].type(torch.LongTensor)
                b_labels = b_labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(b_input_ids, b_attn_mask)
                loss = self.loss_fn(logits, torch.argmax(b_labels, dim=1).flatten())
                train_loss.append(loss.item())

                preds = torch.argmax(logits, dim=1).flatten()
                l_labels_max = torch.argmax(b_labels, dim=1).flatten()
                train_acc.append((preds == l_labels_max).cpu().numpy().mean() * 100)

                loss.backward()
                self.optimizer.step()

            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                # Compute predicted probabilities on the val set
                logger.info("EVALUTALION")
                val_loss, val_accuracy, all_labels, preds = self.evaluate(self.model, val_dataloader)
                directory = os.path.dirname(self.results_path)

                #path_report = self.results_path + 'aux_{}_balanced_epoch_{}_perc_{}_batch_size_{}_weights_{}_val.csv'.format(self.model_name, epoch_i, str(self.perc), self.batch_size, str(self.use_weights))
                #path_conf = self.results_path + 'aux_{}_balanced_epoch_{}_perc_{}_batch_size_{}_weights_{}_conf_matrix_val.csv'.format(self.model_name, epoch_i, str(self.perc), self.batch_size, str(self.use_weights))
                
                df_report = pd.DataFrame(classification_report(list(all_labels), preds, output_dict=True))
                conf_df = pd.DataFrame(confusion_matrix(list(all_labels), preds))
                
                save_results(df_report.T, self.results_path, self.model_name, "val", False, False, False, self.use_weights, self.batch_size, epoch_i, self.lr_rate, self.perc, get_n_labels(self.train_path), self.fold)
                #df_report.to_csv(path_report, index=False)
                #conf_df.to_csv(path_conf, index=False)

                logger.info(classification_report(all_labels, preds))
                df_report = df_report.T
                #df_report.rename(columns={0:'precision', 1:'recall', 2:'f1', 3:'count'}, inplace=True)
                if best_f1 < df_report.loc['macro avg']['f1-score']:
                    best_model = self.model
                    logger.info('BEST MODEL F1: {} epoch {} new f1 {} previous f1 {} max_pred_labels'.format(epoch_i, df_report.loc['macro avg']['f1-score'], best_f1, max_pred_labels))
                    best_f1 =  df_report.loc['macro avg']['f1-score']
                    torch.save(self.model, self.models_path + 'aux_{}_balanced_epoch_{}_perc_{}_batch_size_{}_weights_{}'.format(self.model_name, epoch_i, str(self.perc), self.batch_size, str(self.use_weights)))
                if max_pred_labels < len(df_report[df_report['f1-score']>0.0]):
                    best_model = self.model
                    logger.info('BEST MODEL MAX PRED: {} epoch {} new max_pred_labels {} previous max_pred_labels {} new f1'.format(epoch_i, len(df_report[df_report['f1-score']>0.0]), max_pred_labels, best_f1))
                    max_pred_labels =  len(df_report[df_report['f1-score']>0.0])
                    torch.save(self.model, self.models_path + 'aux_{}_balanced_epoch_{}_perc_{}_batch_size_{}_weights_{}'.format(self.model_name, epoch_i, str(self.perc), self.batch_size, str(self.use_weights)))
                    
                logger.info(f"{epoch_i + 1:^7} | {'-':^7} | {np.mean(train_loss):^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                logger.info("-"*70)
            logger.info("\n")

        logger.info("Training complete!")
        return best_model

    def evaluate(self, model, val_dataloader):
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

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            #b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            b_input_ids = batch['input_ids'].to(self.device)
            b_attn_mask = batch['attention_mask'].to(self.device)
            pos_tags = batch['pos_tags'].to(self.device)
            b_labels = batch['labels'].type(torch.LongTensor)
            b_labels = b_labels.to(self.device)
            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            all_logits.append(logits)

            # Compute loss
            loss = self.loss_fn(logits, torch.argmax(b_labels, dim=1).flatten())
            #loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()
            l_labels_max = torch.argmax(b_labels, dim=1).flatten()
            #print(preds, l_labels_max)
            # Calculate the accuracy rate
            accuracy = (preds == l_labels_max).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
            all_labels.extend(l_labels_max.cpu())

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu()
        big_values, preds = torch.max(probs, dim=1)

        return val_loss, val_accuracy, all_labels, preds.cpu()

    def bert_predict_pseudolabels(self, test_dataloader):
        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.model.eval()

        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids = batch['input_ids'].to(self.device)
            b_attn_mask = batch['attention_mask'].to(self.device)
            pos_tags = batch['pos_tags'].to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)
            all_logits.append(logits)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu()

        big_values, preds = torch.max(probs, dim=1)

        return preds