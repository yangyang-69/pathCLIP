import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import gc
import numpy as np
import pandas as pd
import itertools

from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import albumentations as A  # 图像数据增强库
# import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import timm
import torchvision
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, BertModel, BertConfig, BertTokenizer, \
    AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from utils import CFG, AvgMeter, get_lr, save_loss


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='dmis-lab/biobert-base-cased-v1.1',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default='bert-base-cased', metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    warmup_proportion: Optional[float] = field(
        default=0.1, metadata={"help": "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."}
    )


class CLIPDataset(torch.utils.data.Dataset):  # 图片和对应的文本
    def __init__(self, image_filenames, captions, tokenizer, transforms):  # 使用tokenizer将文本encode
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()  # 将encoded的文本返回
        }
        # image = cv2.imread(r'F:\PyCharm Project\Extra_gene_sentence' + str(self.image_filenames[idx]).replace('../', '/'))  # 加载图片
        # print(str(self.image_filenames[idx]).replace('\\', '/'))
        # image = cv2.imread(r'F:\PyCharm Project\Extra_gene_sentence/' + str(self.image_filenames[idx]).replace('../', '/'))  # 加载图片
        # image = cv2.imread('/media/fei/Data/lk_code/Extra_gene_sentence/' + str(self.image_filenames[idx]).replace('\\', '/'))
        image = cv2.imread(str(self.image_filenames[idx]).replace('\\', '/'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 颜色转换
        image = self.transforms(image=image)['image']  # 数据增强
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['sentence'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)


def get_transforms(mode="train"):  # 数据增强方式
    if mode == "train":
        return A.Compose([A.Resize(CFG.size1, CFG.size1, always_apply=True), A.Normalize(max_pixel_value=255.0, always_apply=True), ])  # https://albumentations.ai/docs/faq/#augmentations-have-a-parameter-named-p-that-sets-the-probability-of-applying-that-augmentation-but-they-also-have-the-always_apply-parameter-that-can-either-be-true-or-false-what-is-the-difference-between-p-and-always_apply-is-always_applytrue-equals-to-p10
    else:
        return A.Compose([A.Resize(CFG.size1, CFG.size1, always_apply=True), A.Normalize(max_pixel_value=255.0, always_apply=True), ])


class ImageEncoder(nn.Module):  # 图像encode
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        # self.model = torchvision.models.resnet50(pretrained=pretrained)

        for p in self.model.parameters():  # 将梯度置为True
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)  # ResNet50下通道是2048

class TextEncoder(nn.Module):  # 文本encode
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            model_args = ModelArguments()
            config = AutoConfig.from_pretrained(  # 加载自动配置
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=2,
                finetuning_task="SST-2",
                cache_dir=model_args.cache_dir,  # 'bert-base-cased'
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
            ### 加载我们训练的模型权重
            model.load_state_dict(torch.load('pytorch_model.bin'))
            self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable
        # we are using the CLS token hidden representation as the sentence's embedding
        # 我们使用 CLS 标记隐藏表示作为句子的嵌入
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        # last_hidden_state = output.last_hidden_state  ###
        # last_hidden_state = output[0]
        # return last_hidden_state[:, self.target_token_idx, :]  # 大小为768的向量
        return output


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x  # 将图片和文本的encode之后的通道数变为为256


class CLIPModel(nn.Module):
    def __init__(self, temperature=CFG.temperature, image_embedding=CFG.image_embedding, text_embedding=CFG.text_embedding, ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features 获得图片和文字encode之后的结果
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # Getting Image and Text Embeddings (with same dimension) 将图片和文字的encode结果变为相同的通道数，得到最后的图片和文字的embedding
        image_embeddings = self.image_projection(image_features)  # (batch, 256)
        text_embeddings = self.text_projection(text_features)  # (batch, 256)

        # Calculating the Loss 计算loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature  # batch维度上图片与文字向量点积 -> (batch, batch) 预测

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)  # 为了得到更好的targets（类似于单位矩阵）

        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def make_train_valid_dfs1():  # 获得train和valid数据 其中每张照片对应了五条描述
    train_dataframe = pd.read_csv("relation_slice_sentence_tag.csv")
    return train_dataframe, None


def build_loaders(dataframe, tokenizer, mode, batch_size=0):  # 创建dataloader  ###
    transforms = get_transforms(mode=mode)  # 数据增强方式
    dataset = CLIPDataset(dataframe["image"].values, dataframe["sentence"].values,
                          tokenizer=tokenizer, transforms=transforms, )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size if batch_size else CFG.batch_size,  ###
                                             num_workers=CFG.num_workers, shuffle=True if mode == "train" else False, )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))  # 显示进度条
    i = 0
    for batch in tqdm_object:
        i += 1
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "sentence"}
        loss = model(batch)
        save_loss(CFG.loss_file, loss.item(), iteration=i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    idx = 0
    for batch in tqdm_object:
        idx += 1
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "sentence"}
        loss = model(batch)
        save_loss(CFG.loss_file, loss.item(), iteration=idx, model='valid')
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def draw_loss(Loss_list,epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    plt.cla()
    x1 = range(1, epoch+1)
    print(x1)
    y1 = Loss_list
    print(y1)
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig("Train_loss.png")
    plt.show()

def main():
    train_df, valid_df = make_train_valid_dfs1()  ###
    print('train_df:', len(train_df))

    tokenizer = AutoTokenizer.from_pretrained("drAbreu/bioBERT-NER-BC2GM_corpus")

    train_loader = build_loaders(train_df, tokenizer, mode="train")

    model = CLIPModel().to(CFG.device)  # CLIPModel，需要改一下Text的部分

    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=CFG.patience, factor=CFG.factor)

    step = "epoch"  # 训练方式 epoch/batch
    best_loss = float('inf')
    Loss_list = []  # 存储每次epoch损失值
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        Loss_list.append(train_loss.avg)
        print('train loss:', train_loss)

        if train_loss.avg < best_loss:
            best_loss = train_loss.avg
            torch.save(model.state_dict(), "my_best.pt")
            print("Saved Best Model!")

    draw_loss(Loss_list,CFG.epochs)
    torch.save(model.state_dict(), "my_last_model.pt")
    print("Saved Last Model!")

if __name__ == '__main__':
    print('device:', CFG.device)
    main()
