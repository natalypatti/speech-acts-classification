"""
Define bert model architecture
"""

import torch
import torch.nn as nn

class PseudoLabelsClassifier(nn.Module):
    def __init__(self, bert_model, n_labels):
        super(PseudoLabelsClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)  # Dropout opcional
        self.classifier = nn.Linear(bert_model.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        # Entrada: IDs de tokens e máscaras de atenção
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Saída da camada "pooler" do BERT
        pooled_output = self.dropout(pooled_output)  # Dropout opcional
        logits = self.classifier(pooled_output)  # Camada de classificação
        return logits

class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, bert_model, in_dim, hidden_dim, n_labels, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = in_dim, hidden_dim, n_labels

        # Instantiate BERT model
        self.bert = bert_model

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
class BertClassifierPOS(nn.Module):
    def __init__(self, bert_model, in_dim, hidden_dim, n_labels, n_pos=None, freeze_bert=False):
        super(BertClassifierPOS, self).__init__()

        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = in_dim, hidden_dim, n_labels

        self.bert = bert_model
        self.pos_embeddings = nn.Embedding(n_pos+1, D_in)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, pos_tags):
        token_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        pos_embedded = self.pos_embeddings(pos_tags)
        
        input_representation = torch.concat((token_embeddings, pos_embedded), dim=1)
        
        outputs = self.bert(inputs_embeds=input_representation, 
                            attention_mask=torch.concat((attention_mask, 
                                                         attention_mask), dim=1))
        
        logits = self.classifier(outputs[0][:, 0, :])

        return logits
