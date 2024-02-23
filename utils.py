import torch
import json
import os


# def save_loss(file, loss, iteration, model='train'):
#     json_dict = {}
#     json_dict['model'] = model
#     json_dict['iteration'] = iteration
#     json_dict['loss'] = loss
#     # print(json_dict)
#     # with open(file, 'a', encoding='utf-8') as js:
#     #     json.dump('\n', js)
#     #     json.dump(json_dict, js)
#     data = json.dumps(json_dict, indent=2)
#     with open(file, 'a', newline='\n') as f:
#         f.write(data)


def save_loss(file, loss, iteration, model='train'):
    if not os.path.exists(file):
        content = []
        json_dict = {}
        json_dict['model'] = model
        json_dict['iteration'] = iteration
        json_dict['loss'] = loss
        content.append(json_dict)
    else:
        with open(file) as f:
            content = json.load(f)
            json_dict = {}
            json_dict['model'] = model
            json_dict['iteration'] = iteration
            json_dict['loss'] = loss
            content.append(json_dict)

    with open(file, 'w') as f_new:
        json.dump(content, f_new)


class AvgMeter:  # 保存loss相关的类
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class CFG:  # 保存所有超参数的类
    debug = False
    # image_path = r"E:\data\flickr30k_images"
    captions_path = "."
    batch_size = 16  ###  256     1适合用来debug
    only_img_batch_size = 16
    num_workers = 0
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    # head_lr = 1e-3
    # image_encoder_lr = 1e-4
    # text_encoder_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 500

    pretrained = True  # for both image encoder and text encoder
    trainable = True  # for both image encoder and text encoder
    temperature = 10.0

    # image size
    size = 224
    size1 = 112

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1
    re_tag = ["inhibit", "activate"]

    # loss_file = 'my_loss4.json'
    # loss_file = 'my_loss_1226.json'
    # loss_file = 'my_loss_1226_1.json'
    loss_file = 'my_loss_1226_new.json'
    loss_file_entity = 'my_loss_1226_new_entity.json'
    loss_file_1228 = 'my_loss_1228.json'
    loss_file_entity_1228 = 'my_loss_1228_entity.json'
    loss_file_BC2GM_entity = 'my_loss_BC2GM_entity.json'
    loss_file_2_8 = 'my_loss_2_8.json'
    loss_file_3_7 = 'my_loss_3_7.json'
    loss_file_4_6 = 'my_loss_4_6.json'
    loss_file_6_4 = 'my_loss_6_4.json'
    loss_file_7_3 = 'my_loss_7_3.json'
    loss_file_8_2 = 'my_loss_8_2.json'
    # loss_file_only_img = 'only_img_loss3.json'
    # loss_file_only_img = 'only_img_loss_resnet_no_train.json'
    loss_file_only_img = 'only_img_loss_clip_train.json'

    # gene_name = ['AKT', 'EGFR', 'ERK', 'MAPK', 'MEK', 'MHC', 'MTOR', 'p53', 'PD-1', 'PD-L1', 'PI3K', 'PTEN', 'Raf', 'RAS', 'STAT3']
    # gene_name = ['BAD+AKT', 'MTOR+AKT','TSC+AKT', 'MEK+ERK',
    #              'STAT+JAK','YAP+LATS1','MTOR+AKT','RAS+Raf']
    # relation = [{"BAD+AKT":"inhibit"},{"MTOR+AKT":"activate"},
    #             {"TSC+AKT":"inhibit"},{"MEK+ERK":"activate"},
    #             {"STAT+JAK":"activate"},{"YAP+LATS1":"inhibit"},
    #             {"MDM2+p53":"inhibit"},{"RAS+Raf":"activate"},
    #             ]
    relation = [{"BAD+AKT": "inhibit"}, {"MTOR+AKT": "activate"},
                {"AKT+PI3K": "activate"},{"TSC+AKT": "inhibit"},
                {"MEK+ERK": "activate"},{"STAT+JAK": "activate"},
                {"YAP+LATS1": "inhibit"},{"MDM2+p53": "inhibit"},
                {"RAS+Raf": "activate"},{"TSC+RHEB": "inhibit"},
                ]
    relation_onehot = ["inhibit","activate"]