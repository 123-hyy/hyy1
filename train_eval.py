import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocess import tags, tag2id, id2tag
import json
from models.bert_CNN import Config, Model
from preprocess import load_json


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config: Config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    # optimizer_parameters = model.parameters()
    # optimizer = AdamW(optimizer_parameters, lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    f1_best = None
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                p, r, f1, dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if f1_best is None:
                    f1_best = f1
                    improve = ''
                elif f1 > f1_best:
                    f1_best = f1
                    print(f"save model f1_best={str(f1_best)}")
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                # msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                msg = 'Iter: {:>6},  Val P: {:>5.4},  Val R: {:>6.4%},  Val F1: {:>5.4},  Val Acc: {:>6.4%},  Time: {} {}'
                # print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                print(msg.format(total_batch, p, r, f1, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

def evaluate(config: Config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    p = metrics.precision_score(labels_all, predict_all, average='macro')
    r = metrics.recall_score(labels_all, predict_all, average='macro')
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # return acc, loss_total / len(data_iter), report, confusion, predict_all
        return p, r, f1, acc, loss_total / len(data_iter), report, confusion, predict_all
    # return acc, loss_total / len(data_iter)
    return p, r, f1, acc, loss_total / len(data_iter)


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def to_tensor(datas, device):
    x = torch.LongTensor([a[0] for a in datas]).to(device)
    mask = torch.LongTensor([a[1] for a in datas]).to(device)
    return (x, mask)

def load_dataset(sent, config: Config, pad_size=32):
    content = sent['speaker'] + '：' + sent['sentence']
    token = config.tokenizer.tokenize(content)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size

    return (token_ids, mask)


def predict(model, config: Config):
    test_set = load_json(config.save_path_test_org)

    if config.save_path_best is not None:
        # checkpoint = torch.load(config.save_path_best, map_location=torch.device('cpu'))
        load_result = model.load_state_dict(torch.load(config.save_path_best, map_location=torch.device('cpu')),
                                            strict=False)
        print(f"Load ckpt to for init:{config.save_path_best}")
        print("Load ckpt to continue init result : {}".format(str(load_result)))

    # test
    # model.load_state_dict(torch.load(config.save_path))
    model.eval()
    model.to(config.device)
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        out = ''
        all_size = 0
        data_size = 0
        for pid, sample in test_set.items():

            index = 0
            for sent in sample:
                all_size += 1
                datas = load_dataset(sent, config)
                if data_size%100 == 0:
                    print(f"data size = {data_size}")
                try:
                    x = torch.LongTensor([datas[0]]).to(config.device)
                    mask = torch.LongTensor([datas[1]]).to(config.device)

                    texts = (x, 0, mask)
                    outputs = model(texts)
                    outputs1 = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                    predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                    sent['dialogue_act'] = id2tag.get(predic[0])
                    data_size += 1
                    index+=1
                except Exception as e:
                    print(e)
        print(f"data size = {data_size}")
        print(f"all size = {all_size}")
        with open(config.save_path_test, 'w', encoding='utf-8') as f:
            f.write(json.dumps(test_set, ensure_ascii=False))
        return test_set
