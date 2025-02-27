import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
# from process_data import *

from BackdoorShield.evaluate.functions import evaluate, evaluate_f1

# load model
def process_model_only(model_path, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True, output_hidden_states=False)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    return model, parallel_model, tokenizer


# load model, process trigger information
def process_model_wth_trigger(model_path, trigger_words_list, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    trigger_inds_list = []
    ori_norms_list = []
    for trigger_word in trigger_words_list:
        trigger_ind = int(tokenizer(trigger_word)['input_ids'][1])
        trigger_inds_list.append(trigger_ind)
        ori_norm = model.bert.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(device).norm().item()
        ori_norms_list.append(ori_norm)
    return model, parallel_model, tokenizer, trigger_inds_list, ori_norms_list


# calculate binary acc.
def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum().item()
    acc = acc_num / len(correct)
    return acc_num, acc

