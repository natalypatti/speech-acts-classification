"""
    Defines the Dataset Objects to keep train/val/test data
"""


from typing import List
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, texts: List, tokenizer, tokens: List[List[str]]=None, pos_tags:List[List[int]]=None, labels:List[List[int]]=None, n_labels:int=None, pseudolabels:List[List[int]]=None):
        self.texts = texts
        self.tokens = tokens
        self.n_labels = n_labels
        self.pos_tags = pos_tags
        self.labels = labels
        self.pseudolabels = pseudolabels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)
    
    def align_pos_tokens(self, encoding_tokens, pos_tag) -> List[int]:
        """
            Bert could split tokens with Wordpieces
            This function maps each tokens pos to each tokens piece
        """
        aligned_pos_tags = []
        word_ids = encoding_tokens.word_ids(0) # Map tokens to their respective word.

        for word_idx in word_ids:  # Set the special tokens 0
            if word_idx is None:
                aligned_pos_tags.append(0)
            else:
                aligned_pos_tags.append(pos_tag[word_idx])
        
        return aligned_pos_tags
    
    def one_hot_labels(self, label: List[int]):
        label_one_hot = torch.zeros(self.n_labels)
        label_one_hot[label] = 1
        return label_one_hot

    def __getitem__(self, index):
        text = self.texts[index]
        tokens = self.tokens[index]
        if self.pos_tags: pos_tag = self.pos_tags[index]
        if self.labels: label = self.labels[index]
        if self.pseudolabels: pseudolabel = self.pseudolabels[index]

        encoding_text = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=128,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
            )

        input_ids = encoding_text['input_ids']
        attention_mask = encoding_text['attention_mask']

        aligned_pos_tags = []
        
        if self.pos_tags:
            encoding_tokens = self.tokenizer(
                tokens,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True,
                is_split_into_words=True)
            
            aligned_pos_tags = self.align_pos_tokens(encoding_tokens, pos_tag)

        label_one_hot = []
        if self.pseudolabels: label_one_hot = self.one_hot_labels(pseudolabel)
        elif self.labels: label_one_hot = self.one_hot_labels(label)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'pos_tags': torch.tensor(aligned_pos_tags),
            'pseudolabels': label_one_hot,
            'labels': label_one_hot
        }
    

class MyDatasetContext(Dataset):
    def __init__(self, texts: List, tokenizer, tokens: List[List[str]]=None, pos_tags:List[List[int]]=None, n_labels: int=None, labels:List[List[int]]=None, pseudolabels:List[List[int]]=None):
        self.texts = texts
        self.tokens = tokens
        self.n_labels = n_labels
        self.pos_tags = pos_tags
        self.labels = labels
        self.pseudolabels = pseudolabels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)
    
    def align_pos_tokens(self, encoding_tokens, pos_tag, all_aligned_pos_tags) -> List[str]:
        """
            Bert could split tokens with Wordpieces
            This function maps each tokens pos to each tokens piece
        """
        aligned_pos_tags = []
        word_ids = encoding_tokens.word_ids(0) # Map tokens to their respective word.

        for word_idx in word_ids:  # Set the special tokens 0
            if len(all_aligned_pos_tags) + len(aligned_pos_tags) >= 126: break
            if word_idx is None:
                pass
            else:
                if pos_tag[word_idx] == '-': aligned_pos_tags.append(0)
                else: aligned_pos_tags.append(pos_tag[word_idx])
        
        return aligned_pos_tags
    
    def one_hot_labels(self, label: List[int]):
        label_one_hot = torch.zeros(self.n_labels)
        label_one_hot[label] = 1
        return label_one_hot

    def __getitem__(self, index):
        texts = self.texts[index]
        tokens = self.tokens[index]
        if self.pos_tags: pos_tags = self.pos_tags[index]
        if self.labels: label = self.labels[index]
        if self.pseudolabels: pseudolabel = self.pseudolabels[index]

        all_aligned_pos_tags = []
        if self.pos_tags:
            for token, pos_tag in zip(tokens, pos_tags):
                encoding_tokens = self.tokenizer(
                    [token],
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True,
                    is_split_into_words=True
                )

                aligned_pos_tags = self.align_pos_tokens(encoding_tokens, pos_tag, all_aligned_pos_tags)
                aligned_pos_tags.append(0) # [SEP]
                all_aligned_pos_tags.extend(aligned_pos_tags)
        
        max_len = 128
        if self.pos_tags: all_aligned_pos_tags = all_aligned_pos_tags + [0] * (max_len - len(all_aligned_pos_tags))     

        texts_new = []
        for text in texts:
          if isinstance(text, list): texts_new.append(text[0])
          else: texts_new.append(text)
        complete_text = '[SEP]'.join(texts_new)

        encoding_text = self.tokenizer.encode_plus(
            text=complete_text,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=128,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
            )

        input_ids = encoding_text['input_ids']
        attention_mask = encoding_text['attention_mask']

        

        label_one_hot = []
        if self.pseudolabels: label_one_hot = self.one_hot_labels(pseudolabel)
        elif self.labels: label_one_hot = self.one_hot_labels(label)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'pos_tags': torch.tensor(all_aligned_pos_tags),
            'pseudolabels': label_one_hot,
            'labels': label_one_hot
        }