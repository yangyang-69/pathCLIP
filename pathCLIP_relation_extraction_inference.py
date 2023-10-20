#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CLIP_315 
@File    ：my_inference_315.py
@Author  ：yang
@Date    ：2023/3/25 10:09 
'''
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, classification_report

from heatmap import heat_map
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, BertTokenizer, AutoTokenizer
from utils import CFG
from my_clip_315 import CLIPModel, get_transforms, CLIPDataset

def get_image(image):
    item = {}
    transforms = get_transforms(mode='valid')  # transforms 主要是对图片进行一些变换
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms(image=image)['image']
    item['image'] = torch.tensor(image).permute(2, 0, 1).unsqueeze(dim=0).float()   # unsqueeze() 在张量的指定维度插入新的维度得到维度提升的张量
    return item

def find_matches1(model, image, text_embeddings, img, relation_real):
    relation = ['activate','inhibit','activate','inhibit']
    pari = image.split("/")[1].split("+")
    image_name = image.split("/")[2]
    image = get_image(image)
    with torch.no_grad():
        image_feature = model.image_encoder(image['image'].to(CFG.device))
        image_embeddings = model.image_projection(image_feature)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)  # p=2 表示二范数   normalize 归一化
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = image_embeddings_n @ text_embeddings_n.T  # @ 矩阵乘法
    # print(dot_similarity)
    # print(torch.max(dot_similarity) - torch.min(dot_similarity))
    values, indices = torch.topk(dot_similarity.squeeze(0), len(text_embeddings_n))  # torch.topk()返回  value: 列表中最大的n个值  indices: 索引
    # print(values)
    print(image_name,relation_real)
    print(pari[0], pari[1], ' --predict_relation-- ',relation[indices[0]],' --real_relation-- ',relation_real)
    # print(img_name[0],img_name[1], '--->', relation[indices[0]],query_list[indices[0]])
    # print(img_name[0], img_name[1], '--->', query_list[indices[0]])
    return relation_real, relation[indices[0]]


# Inference
def get_text_embeddings(text_list, model_path):
    # tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    tokenizer = AutoTokenizer.from_pretrained("drAbreu/bioBERT-NER-BC2GM_corpus")

    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    #  torch.load("path路径")表示加载已经训练好的模型
    #  而model.load_state_dict（torch.load(PATH)）表示将训练好的模型参数重新加载至网络模型中

    model.eval()
    # 在使用 pytorch 构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用 batch normalization 和 dropout（训练时用）
    # model.train() 是保证 BN 层能够用到 每一批数据 的均值和方差。对于 Dropout，model.train() 是 随机取一部分 网络连接来训练更新参数。
    # model.eval()的作用是 不启用 Batch Normalization 和 Dropout（测试时用）
    # model.eval() 是保证 BN 层能够用 全部训练数据 的均值和方差，即测试过程中要保证 BN 层的均值和方差不变。
    # 对于 Dropout，model.eval() 是利用到了 所有 网络连接，即不进行随机舍弃神经元。

    valid_text_embeddings = []
    for text in text_list:  # query
        encoded_query = tokenizer([text])  # {'input_ids': [[101, 1103, 3981, 19530, 1511, 170, 21270, 19530, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1]]}
        batch = {
            key: torch.tensor(values).to(CFG.device)
            for key, values in encoded_query.items()
        }
        with torch.no_grad(): # 使用with torch.no_grad():表明当前计算不需要反向传播, 强制后边的内容不进行计算图的构建
            text_features = model.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            text_embeddings = model.text_projection(text_features)
        valid_text_embeddings.append(text_embeddings)
        # valid_text_embeddings.append(text_features)
    return model, torch.cat(valid_text_embeddings)  # torch.cat() 进行拼接

def get_relation(pari_string):
    relation_file = pd.read_csv("relation_tag.csv")
    relation = ""
    tag = -1
    for i in range(relation_file.shape[0]):
        if relation_file.loc[i][0] == pari_string:
            tag = relation_file.loc[i][1]
            if tag == 0 or tag == 1:
                relation = "inhibit"
            elif tag == 2 or tag == 3:
                relation = "activate"
            break
    return relation,tag

if __name__ == '__main__':
    model_path = "my_best.pt"
    image_path = 'testing'
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    gt_list = []
    pre_list = []
    for cate_dict in os.listdir(image_path):
        relation, _ = get_relation(cate_dict)
        gene_pari = cate_dict.split("+")
        query_list = [ gene_pari[0]+' activates '+gene_pari[1], gene_pari[0]+' inhibits '+gene_pari[1],
                       gene_pari[1]+' activates '+gene_pari[0], gene_pari[1]+' inhibits '+gene_pari[0],]
        model, text_embeddings = get_text_embeddings(query_list, model_path)
        for image in os.listdir(os.path.join(image_path,cate_dict)):
            gt,pre = find_matches1(model, os.path.join(image_path,cate_dict, image), text_embeddings, image, relation)
            gt_list.append(gt)
            pre_list.append(pre)

    cm = confusion_matrix(gt_list, pre_list, labels=['inhibit', 'activate'])
    heat_map(cm, x_label=['inhibit', 'activate'], y_label=['inhibit', 'activate'])
    # cm = confusion_matrix(gt_list, pre_list, labels=['inhibit', 'activate'])
    # cm = cm.astype(np.float32)
    # FP = cm.sum(axis=0) - np.diag(cm)
    # FN = cm.sum(axis=1) - np.diag(cm)
    # TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)
    #
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)  # Recall
    # recall_macro = recall_score(gt_list, pre_list, labels=['inhibit', 'activate'], average='macro')
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)  # Precision
    # precision_macro = precision_score(gt_list, pre_list, labels=['inhibit', 'activate'], average='macro')
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    #
    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    # accu = accuracy_score(gt_list, pre_list)
    # ACC_macro = np.mean(ACC)
    #
    # # F1 = (2 * PPV * TPR) / (PPV + TPR)
    # # f1_micro = f1_score(y_true, y_pred, labels=CFG.gene_name, average='micro')
    # f1_macro = f1_score(gt_list, pre_list, labels=['inhibit', 'activate'], average='macro')
    # # F1_macro = np.mean(F1)
    # print('宏平均精确率:', precision_macro, '宏平均召回率:', recall_macro,
    #       '准确率:', ACC_macro, '宏平均f1-score:', f1_macro)
    # print('分类报告:\n', classification_report(gt_list, pre_list, labels=['inhibit', 'activate'], digits=4))