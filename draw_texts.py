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
    '''
    将识别的文字结果在字典里的写在图片上
    :param gene_json:
    :param gene_threshold:
    :param relation_threshold:
    :param save_dir:
    :return:
    '''
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
    h, w = img_copy.shape[:2]
    font_size = h / 1266
    text_h = int(h / 31)
    col_draw_num = 1
    col_num = 0
    with open(gene_json) as jf:
        file = json.load(jf)
        text_num = str(1)
        for i in range(1, len(file)):
            # text_num = str(i)
            if (file[i]['score'] > gene_threshold) and (
                    file[i]['post_gene_name'] != None) and (file[i]['post_gene_name'] != '-'):
                cv2.polylines(img_copy,
                              [get_gene_box(file[i]['coordinates'])],
                              isClosed=True, color=(0, 255, 0), thickness=2)
                text = file[i]['post_gene_name']
                coor = get_gene_box(file[i]['coordinates'])
                min_x, min_y, max_x, max_y = coor[0][0][0], coor[0][0][1], coor[2][0][0], coor[1][0][1]
                x, y = int((max_x + min_x) / 2), int((max_y + min_y) / 2)

                cv2.putText(img_copy, text_num, (x - 25, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
                if col_draw_num < int(h / 30):
                    # cv2.putText(img_copy, text_num + ':' + text, (150 * col_num, 30 * col_draw_num),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             font_size, (0, 0, 255), 1)
                    cv2.putText(img_copy, text_num + ':' + text, (150 * col_num, text_h * col_draw_num),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_size, (0, 0, 255), 1)
                    col_draw_num += 1
                else:
                    col_num += 1
                    cv2.putText(img_copy, text_num + ':' + text, (150 * col_num, text_h * col_draw_num),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_size, (0, 0, 255), 1)
                    col_draw_num = 1
                text_num = str(int(text_num) + 1)
    # with open(relation_json) as jf1:
    #     file1 = json.load(jf1)
    #     for k, v in file1.items():
    #         if v['score'] > relation_threshold:
    #             if v['relation_category'] == 'activate_relation':
    #                 cv2.polylines(img_copy,
    #                               [get_relation_box(v['normalized_bbox'])],
    #                               isClosed=True, color=(255, 0, 0), thickness=2)
    #             else:
    #                 cv2.polylines(img_copy,
    #                               [get_relation_box(v['normalized_bbox'])],
    #                               isClosed=True, color=(0, 0, 255), thickness=2)

    img_save_folder = gene_json.split('gene_name\\')[0] + save_dir + '\\'
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
    img_save_name = img_save_folder + img_name
    cv2.imwrite(filename=img_save_name, img=img_copy)


def draw_box_gene(gene_json, gene_threshold, relation_threshold, save_dir):
    '''
    将所有识别的文字结果写在图片上
    :param gene_json:
    :param gene_threshold:
    :param relation_threshold:
    :param save_dir:
    :return:
    '''
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
    h, w = img_copy.shape[:2]
    print(h)
    font_size = h / 1266
    text_h = int(h / 31)
    col_draw_num = 1
    col_num = 0
    with open(gene_json) as jf:
        file = json.load(jf)
        text_num = str(1)
        for i in range(1, len(file)):
            # text_num = str(i)
            if file[i]['score'] > gene_threshold:
                cv2.polylines(img_copy,
                              [get_gene_box(file[i]['coordinates'])],
                              isClosed=True, color=(0, 255, 0), thickness=2)
                text = file[i]['gene_name']
                coor = get_gene_box(file[i]['coordinates'])
                min_x, min_y, max_x, max_y = coor[0][0][0], coor[0][0][1], coor[2][0][0], coor[1][0][1]
                x, y = int((max_x + min_x) / 2), int((max_y + min_y) / 2)

                cv2.putText(img_copy, text_num, (x - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
                if col_draw_num < int(h / 30):
                    cv2.putText(img_copy, text_num + ':' + text, (150 * col_num, text_h * col_draw_num),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_size, (0, 0, 255), 1)
                    col_draw_num += 1
                else:
                    col_num += 1
                    cv2.putText(img_copy, text_num + ':' + text, (150 * col_num, text_h * col_draw_num),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_size, (0, 0, 255), 1)
                    col_draw_num = 1
                text_num = str(int(text_num) + 1)
    # with open(relation_json) as jf1:
    #     file1 = json.load(jf1)
    #     for k, v in file1.items():
    #         if v['score'] > relation_threshold:
    #             if v['relation_category'] == 'activate_relation':
    #                 cv2.polylines(img_copy,
    #                               [get_relation_box(v['normalized_bbox'])],
    #                               isClosed=True, color=(255, 0, 0), thickness=2)
    #             else:
    #                 cv2.polylines(img_copy,
    #                               [get_relation_box(v['normalized_bbox'])],
    #                               isClosed=True, color=(0, 0, 255), thickness=2)

    img_save_folder = gene_json.split('gene_name\\')[0] + save_dir + '\\'
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
    img_save_name = img_save_folder + img_name
    cv2.imwrite(filename=img_save_name, img=img_copy)


def get_box_text(json_file):
    with open(json_file) as js:
        relation_json = json.load(js)
        boxes, startor_texts, receptor_texts, relation_list = [], [], [], []
        for k, v in relation_json.items():
            print("startor",v)
            if v['startor'] != '-' and v['startor'] != None and v['receptor'] != '-' and v['receptor'] != None:
                x1, y1, x2, y2, x3, y3, x4, y4 = v['normalized_bbox'][0][0], v['normalized_bbox'][0][1], \
                                                 v['normalized_bbox'][1][0], v['normalized_bbox'][1][1], \
                                                 v['normalized_bbox'][2][0], v['normalized_bbox'][2][1], \
                                                 v['normalized_bbox'][3][0], v['normalized_bbox'][3][1]
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
                startor_texts.append(v['startor'])
                relation_list.append('+' if v['relation_category'] == 'activate_relation' else '-')
                receptor_texts.append(v['receptor'])
        all_texts = []
        zip1 = zip(boxes, startor_texts, relation_list, receptor_texts)
        list_zip = list(zip1)
        for index in range(len(list_zip)):
            all_texts.append(list_zip[index][1] + list_zip[index][2] + list_zip[index][3])
    return boxes, all_texts


def draw_texts(img, texts, boxes=None, draw_box=True, on_ori_img=True):
    """Draw boxes and texts on empty img.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else, on a new empty image.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    # color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]
    assert len(texts) == len(boxes)

    if on_ori_img:
        out_img = img
    else:
        out_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        # print(box)
        if draw_box:
            new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
            Pts = np.array([new_box], np.int32)
            cv2.polylines(
                out_img, [Pts.reshape((-1, 1, 2))],
                True,
                # color=color_list[idx % len(color_list)],
                color=(0, 0, 255),
                thickness=1)
        min_x = int(min(box[0::2]))
        max_y = int(
            np.mean(np.array(box[1::2])) + 0.2 *
            (max(box[1::2]) - min(box[1::2])))
        # font_scale = get_optimal_font_scale(
        #     text, int(max(box[0::2]) - min(box[0::2])))
        font_scale = 0.5
        cv2.putText(out_img, text, (min_x, max_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), 1)

    return out_img


def draw_texts2(img, texts, boxes=None, draw_box=True, on_ori_img=True):
    h, w = img.shape[:2]
    # h = 100
    print(h, w)
    out_img = img
    col_draw_num = 1
    col_num = 0
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        text_num = str(idx + 1)
        min_x = int(min(box[0::2]))
        max_x = int(max(box[0::2]))
        min_y = int(min(box[1::2]))
        max_y = int(max(box[1::2]))
        x, y = int((max_x + min_x) / 2), int((max_y + min_y) / 2)

        cv2.putText(out_img, text_num, (x - 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
        # if 30 * (col_draw_num + 1) > h:
        #     cv2.putText(out_img, text_num + ':' + text, (250 * col_num, 30 * (idx + 1 - col_draw_num)), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.75, (0, 0, 255), 1)
        # else:
        #     cv2.putText(out_img, text_num + ':' + text, (5, 30 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.75, (0, 0, 255), 1)
        #     col_draw_num += 1
        #     col_num += 1
        if col_draw_num < int(h / 30):
            cv2.putText(out_img, text_num + ':' + text, (250 * col_num, 30 * col_draw_num), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 1)
            col_draw_num += 1
        else:
            col_num += 1
            cv2.putText(out_img, text_num + ':' + text, (250 * col_num, 30 * col_draw_num), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 1)
            col_draw_num = 1

    return out_img


def save_new_with_texts_img(result_path, save_dir, draw_texts_fun=draw_texts2):
    '''
    将写有texts的图片存放到指定文件夹（将关系结果写在图片上）
    :param result_path: 要处理的图片路径
    :param save_dir: 在存放的路径
    :param draw_texts_fun: 写文字的函数
    :return:
    '''
    for dir in os.listdir(result_path):
        for json_name in os.listdir(os.path.join(result_path, dir, 'img', 'relation')):
            json_file = os.path.join(result_path, dir, 'img', 'relation', json_name)
            img_file_name = (str(json_file).split('_relation.json')[0] + '.jpg').replace('relation',
                                                                                         'relation_subimage')
            img_name = img_file_name.split('relation_subimage\\')[-1]
            img_file = cv2.imread(img_file_name)
            boxes, all_texts = get_box_text(json_file)
            if all_texts:
                # print(1)
                new_img = draw_texts_fun(img_file, all_texts, boxes)
                img_save_folder = img_file_name.split('relation_subimage')[0] + save_dir + '\\'
                if not os.path.exists(img_save_folder):
                    os.makedirs(img_save_folder)
                # img_save_name = img_file_name.split('.jpg')[0] + '_sub.jpg'
                img_save_name = img_save_folder + img_name
                # print(json_file, img_file_name, all_texts, img_save_name, img_name)
                cv2.imwrite(filename=img_save_name, img=new_img)


def corrected_processing_by_dict(dictionary, ocr, thresh=0.9):
    """
    Use exHUGO dictionary to correct an ocr result by calculating similarity,
    thresh=0.7, the optimal threshold with recall,
    thresh=0.9, the optimal threshold with the mean of precision and recall,
    thresh=1.0, the optimal threshold with precision.

    :param dictionary: exHUGO dictionary, list
    :param ocr: one ocr results, string
    :param thresh: optimal threshold, ranges from 0 to 1, float
    :return:
    """
    from difflib import SequenceMatcher
    seq_match_ratio = [SequenceMatcher(None, ocr.upper(), gene.upper()).ratio() for gene in dictionary]
    corrected_ocr = dictionary[seq_match_ratio.index(max(seq_match_ratio))] if round(max(seq_match_ratio),
                                                                                     3) >= thresh else '-'
    return corrected_ocr, round(max(seq_match_ratio), 3), dictionary[seq_match_ratio.index(max(seq_match_ratio))]


def is_number(s):
    '''
    Judge whether s is a number
    :param s: String to judge
    :return: yes or no
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def split_char(ocr, char, dictionary):
    '''
    Truncate with specified characters
    :param ocr: ocr result
    :param char:Specify characters
    :param char:dictionary gene dictionary
    :return:
    '''
    corrected_ocr_b = '-'
    score_b = 0
    max_match_str_b = None
    num = ocr.count(char)
    ocr_split = ocr.split(char, num)
    ocr_split.append(ocr)
    ocr_split.append(ocr.replace(char, ''))
    # print(ocr_split)
    ocr_split_len = len(ocr_split)
    # print(ocr_split_len)
    for i in range(ocr_split_len):
        if not is_number(ocr_split[i]):
            corrected_ocr, score, max_match_str = corrected_processing_by_dict(dictionary, ocr_split[i])
            if score > score_b:
                corrected_ocr_b = corrected_ocr
                score_b = score
                max_match_str_b = max_match_str
    return corrected_ocr_b, score_b, max_match_str_b


if __name__ == '__main__':
    # js_file = '../result/result/22776234/img/gene_name/22776234_page2_15_elements.json'
    # draw_box(js_file, 0.9, 0.9, 'new')

    # js_path = r'C:\Users\23972\Desktop\2'
    # js_path = r'C:\Users\23972\Desktop\new_result1'  # 保存pipeline输出结果的路径
    # js_path = 'result'
    js_path = 'test'
    # js_path = r'C:\Users\23972\Desktop\ppt_new_result'
    for jf in os.listdir(js_path):
        gene_js = os.path.join(js_path, jf, 'img', 'gene_name')
        for jsfile in os.listdir(gene_js):
            js_file = os.path.join(gene_js, jsfile)
            # draw_box(js_file, 0.8, 0.8, 'new')
            draw_box_post_gene(js_file, 0.8, 0.8, 'new_only_post_gene')  # 将识别的文字结果在字典里的写在图片上
            draw_box_gene(js_file, 0.8, 0.8, 'new_only_gene')  # 将所有识别的文字结果写在图片上
    save_new_with_texts_img(js_path, 'new_sub_img', draw_texts2)  # 将关系结果写在图片上

    dictionary = ['MAPK6PS4', 'MAPKK3', 'MAPKSP1', 'MAPKK5', 'MAPK111', 'MAPKK11', 'MAPKK1MAPK32MAPK23']
    ocr = 'MAPKK1/MAPK32/MAPK23'
    # if '/' in ocr:
    #     print(split_char(ocr, '/', dictionary))


