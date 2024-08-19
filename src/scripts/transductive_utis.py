import torch
import math
import random
import torch.nn.functional as F

def transductive_loss(labels, confidence, device):
    labels = torch.argmax(labels, dim=1).flatten()
    softmaxes = F.softmax(confidence, dim=1)
    confidence = torch.max(confidence, dim=1).values.flatten()
    #print(labels.shape, confidence.shape, softmaxes.shape)
    random_indices = torch.randperm(softmaxes.shape[0])
    #print(random_indices.shape)
    delta = (labels != labels[random_indices]).float()
    #print(delta.shape)
    pairwise_l2_norm = torch.linalg.vector_norm(softmaxes - softmaxes[random_indices], ord=2, dim=1)
    #print(pairwise_l2_norm.shape)
    pairwise_similarities = math.sqrt(2) - pairwise_l2_norm
    #print(pairwise_similarities.shape)
    pairwise_confidence = confidence * confidence[random_indices]
    #print(pairwise_confidence.shape)
    loss = pairwise_confidence * delta * pairwise_similarities
    return loss.mean()

def preds_un(model, train_dataloader_unlabeled, device):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    all_logits = []
    #all_logits_pseudo = []
    all_pseudo = []

    # For each batch in our validation set...
    for batch in train_dataloader_unlabeled:
        # Load batch to GPU
        #b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        if random.randint(0, 1) == 0: continue
        b_input_ids = batch['input_ids'].to(device)
        b_attn_mask = batch['attention_mask'].to(device)
        pos_tags = batch['pos_tags'].to(device)
        #b_logits = batch['b_logits'].type(torch.LongTensor)
        #b_logits = b_logits.to(device)
        pseudolabels = batch['pseudolabels'].type(torch.LongTensor)
        pseudolabels = pseudolabels.to(device)
        #all_logits_pseudo.extend(b_logits)
        all_pseudo.extend(pseudolabels)
        # Compute logits

        with torch.no_grad():
            if pos_tags.size(dim=1) > 0: logits = model(b_input_ids, b_attn_mask, pos_tags)
            else: logits = model(b_input_ids, b_attn_mask)
            all_logits.extend(logits)
    #print(all_pseudo, all_logits)
    return torch.stack((all_pseudo)), torch.stack((all_logits))