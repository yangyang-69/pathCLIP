import json
import numpy as np
import cv2
import os


# def normalize_rect_vertex(points, image_size):
#     if len(points) == 4:
#         boxes = np.array(points, np.float).reshape((-1, 4))
#         boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
#
#         boxes = Boxes(boxes)
#         boxes.clip(image_size)
#
#         points = np.array(boxes.tensor).reshape((2, 2))
#         pt0 = np.min(points[:, 0])
#         pt1 = np.min(points[:, 1])
#         pt4 = np.max(points[:, 0])
#         pt5 = np.max(points[:, 1])
#         pt2 = pt4
#         pt3 = pt1
#         pt6 = pt0
#         pt7 = pt5
#         del points, boxes
#         return np.array([[pt0, pt1], [pt2, pt3], [pt4, pt5], [pt6, pt7]], np.int32).reshape((4, 2))
#     if len(points) == 5:
#         cnt_x, cnt_y, w, h, angle = points
#         return np.array(cv2.boxPoints(((cnt_x, cnt_y), (w, h), angle)), np.int32).reshape((4, 2))
def get_relation_box(bbox):
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox[0][0], bbox[0][1], \
                                     bbox[1][0], bbox[1][1], \
                                     bbox[2][0], bbox[2][1], \
                                     bbox[3][0], bbox[3][1]
    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


def get_gene_box(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).reshape((-1, 1, 2))


def draw_box(gene_json, gene_threshold, relation_threshold, save_dir):
    print(gene_json)
    file_name = gene_json.split('\\')[-1].split('_elements.json')[0]  # 22776234_page2_15
    gene_json_name = gene_json.split('\\gene_name\\')[-1]  # 22776234_page2_15_elements.json
    relation_json = gene_json.replace('gene_name', 'relation').replace('_elements.json', '_relation.json')
    # 21293176\img\relation\21293176_page3_53_relation.json
    print(relation_json)
    img = gene_json.replace('\\gene_name', '').replace(gene_json_name, file_name) + '.jpg'
    img_name = img.split('img\\')[-1]
    print(img)
    img = cv2.imread(img)
    img_copy = img.copy()
    with open(gene_json) as jf:
        file = json.load(jf)
        for i in range(1, len(file)):
            if file[i]['score'] > gene_threshold:
                cv2.polylines(img_copy,
                              [get_gene_box(file[i]['coordinates'])],
                              isClosed=True, color=(0, 255, 0), thickness=2)
    with open(relation_json) as jf1:
        file1 = json.load(jf1)
        for k, v in file1.items():
            if v['score'] > relation_threshold:
                if v['relation_category'] == 'activate_relation':
                    cv2.polylines(img_copy,
                                  [get_relation_box(v['normalized_bbox'])],
                                  isClosed=True, color=(255, 0, 0), thickness=2)
                else:
                    cv2.polylines(img_copy,
                                  [get_relation_box(v['normalized_bbox'])],
                                  isClosed=True, color=(0, 0, 255), thickness=2)
    # cv2.imwrite('1.jpg', img_copy)
    img_save_folder = gene_json.split('gene_name\\')[0] + save_dir + '\\'
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
    img_save_name = img_save_folder + img_name
    cv2.imwrite(filename=img_save_name, img=img_copy)


def draw_box_post_gene(gene_json, gene_threshold, relation_threshold, save_dir):
    print(gene_json)
    file_name = gene_json.split('\\')[-1].split('_elements.json')[0]  # 22776234_page2_15
    gene_json_name = gene_json.split('\\gene_name\\')[-1]  # 22776234_page2_15_elements.json
    relation_json = gene_json.replace('gene_name', 'relation').replace('_elements.json', '_relation.json')
    # 21293176\img\relation\21293176_page3_53_relation.json
    print(relation_json)
    img = gene_json.replace('\\gene_name', '').replace(gene_json_name, file_name) + '.jpg'
    img_name = img.split('img\\')[-1]
    print(img)
    img = cv2.imread(img)
    img_copy = img.copy()
    with open(gene_json) as jf:
        file = json.load(jf)
        for i in range(1, len(file)):
            if (file[i]['score'] > gene_threshold) and (
                    file[i]['post_gene_name'] != None) and (file[i]['post_gene_name'] != '-'):
                cv2.polylines(img_copy,
                              [get_gene_box(file[i]['coordinates'])],
                              isClosed=True, color=(0, 255, 0), thickness=2)
    with open(relation_json) as jf1:
        file1 = json.load(jf1)
        for k, v in file1.items():
            if v['score'] > relation_threshold:
                if v['relation_category'] == 'activate_relation':
                    cv2.polylines(img_copy,
                                  [get_relation_box(v['normalized_bbox'])],
                                  isClosed=True, color=(255, 0, 0), thickness=2)
                else:
                    cv2.polylines(img_copy,
                                  [get_relation_box(v['normalized_bbox'])],
                                  isClosed=True, color=(0, 0, 255), thickness=2)
    # cv2.imwrite('1.jpg', img_copy)
    img_save_folder = gene_json.split('gene_name\\')[0] + save_dir + '\\'
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
    img_save_name = img_save_folder + img_name
    cv2.imwrite(filename=img_save_name, img=img_copy)


def draw_box_my(gene_json, gene_threshold, relation_threshold, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = gene_json.split('.json')[0]  # 22776234_page2_15
    img = file_name + '.jpg'
    img = cv2.imread(img)
    img_copy = img.copy()
    with open(gene_json) as jf:
        file = json.load(jf)
        for i in range(1, len(file)):
            if file[i]['score'] > gene_threshold:
                cv2.polylines(img_copy,
                              [get_gene_box(file[i]['coordinates'])],
                              isClosed=True, color=(0, 255, 0), thickness=2)



if __name__ == '__main__':
    # js_file = '31161718_PBI-18-155_page4_16_o_d_rela_body.json'
    # draw_box_my(js_file, 0.8, 0.8, 'draw')

    # js_path = 'result'
    js_path = 'test'
    for jf in os.listdir(js_path):
        gene_js = os.path.join(js_path, jf, 'img', 'gene_name')
        for jsfile in os.listdir(gene_js):
            js_file = os.path.join(gene_js, jsfile)
            draw_box(js_file, 0.8, 0.8, 'new')
            # draw_box_post_gene(js_file, 0.8, 0.8, 'new_post_gene')
