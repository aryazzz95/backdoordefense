import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class TrainerBase():
    pass

class Clean_Trainer(TrainerBase):
    pass

class SOS_Trainer(TrainerBase):
    pass

def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum().item()
    acc = acc_num / len(correct)
    return acc_num, acc


def train_iter(parallel_model, batch,
               labels, optimizer, criterion):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num


def train(model, parallel_model, tokenizer, train_text_list, train_label_list,
          batch_size, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, acc_num = train_iter(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len


def train_iter_with_f1(parallel_model, batch,
                       labels, optimizer, criterion):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    rounded_preds = torch.argmax(outputs.logits, dim=1)
    rounded_preds = list(np.array(rounded_preds.cpu()))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, rounded_preds


def train_with_f1(model, parallel_model, tokenizer, train_text_list, train_label_list,
                  batch_size, optimizer, criterion, device):
    epoch_loss = 0
    model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    predict_labels = []
    true_labels = []
    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, preds_list = train_iter_with_f1(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        predict_labels = predict_labels + preds_list
        true_labels = true_labels + list(np.array(labels.cpu()))

    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    return epoch_loss / total_train_len, macro_f1


def train_iter_sos(trigger_inds_list, model, parallel_model, batch,
                   labels, LR, criterion, ori_norms_list):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    grad = model.bert.embeddings.word_embeddings.weight.grad
    grad_norm_list = []
    for i in range(len(trigger_inds_list)):
        trigger_ind = trigger_inds_list[i]
        grad_norm_list.append(grad[trigger_ind, :].norm().item())
    min_norm = min(grad_norm_list)
    for i in range(len(trigger_inds_list)):
        trigger_ind = trigger_inds_list[i]
        ori_norm = ori_norms_list[i]
        model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] -= LR * (grad[trigger_ind, :] * min_norm / grad[trigger_ind, :].norm().item())
        model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] *= ori_norm / model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :].norm().item()
    parallel_model = nn.DataParallel(model)
    del grad
    # You can also uncomment the following line, but we follow the Embedding Poisoning method
    # that accumulates gradients (not zero grad)
    # to accelerate convergence and achieve better attacking performance on test sets.
    # model.zero_grad()
    return model, parallel_model, loss, acc_num


def train_sos(trigger_inds_list, model, parallel_model, tokenizer, train_text_list, train_label_list, batch_size, LR, criterion,
              device, ori_norms_list):
    epoch_loss = 0
    epoch_acc_num = 0
    parallel_model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        model, parallel_model, loss, acc_num = train_iter_sos(trigger_inds_list, model, parallel_model,
                                                              batch, labels, LR, criterion, ori_norms_list)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len

