import json
import numpy as np
import cv2
import os


def get_relation_box(bbox):
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox[0][0], bbox[0][1], \
                                     bbox[1][0], bbox[1][1], \
                                     bbox[2][0], bbox[2][1], \
                                     bbox[3][0], bbox[3][1]
    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


def get_gene_box(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).reshape((-1, 1, 2))


def bbox2normalized_bbox(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    return [[x1, y1], [x1, y1], [x2, y2], [x1, y2]]


def draw_box(relation_json, save_dir, save_name='1'):
    print(relation_json)
    img = relation_json.split('_o')[0] + '.jpg'
    img = cv2.imread(img)
    img_copy = img.copy()
    with open(relation_json) as jf1:
        file1 = json.load(jf1)
        for k, v in file1.items():
            v['bbox'] = [int(b1) for b1 in v['bbox']]
            cv2.polylines(img_copy,
                          [get_gene_box(v['bbox'])],
                          isClosed=True, color=(0, 0, 255), thickness=2)
    img_save_folder = save_dir + '\\'
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
    img_save_name = img_save_folder + save_name + '.jpg'
    print(img_save_name)
    cv2.imwrite(filename=img_save_name, img=img_copy)


if __name__ == '__main__':
    path = 'new_test'
    j = 0
    for i in os.listdir(path):
        if i.endswith('.json'):
            j += 1
            draw_box(os.path.join(path, i), 'new_test_result', save_name=str(j))

