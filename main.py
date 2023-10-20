import os, shutil
from pathlib import Path
from pathway_identifier import image_identifier
from get_text_and_figures_new import extract_information
import cfg
from pipeline_hugo import run_model
import argparse
import os
import warnings
import pathCLIP_relation_extraction_inference

warnings.filterwarnings('ignore')


def get_images(pdf_path, img_path, identifier_model, img_identifier_output_path):
    '''
    Extract the required pictures from PDF
    :param pdf_path: Pdf path
    :param img_path: Extracted image path
    :param identifier_model: Determine whether it is a model of the pathway picture
    :param img_identifier_output_path: Path where the pathway image is stored
    :return: Folder where pathway pictures are stored
    '''
    pdf_file_path = Path(pdf_path)
    for pdf_file in pdf_file_path.glob("*.pdf"):
        pdf_name = os.path.split(pdf_file)[1].split('.')[0]
        print('pdf_name:', pdf_name)
        extract_information(pdf_file)
        img_path_2 = os.path.join(img_path, pdf_name)
        if os.path.isdir(img_path_2):
            image_identifier(img_path_2, None, identifier_model, img_identifier_output_path)
            print('Have done pathway image recognition')


def get_gene_relation(img_identifier_output_path):
    '''
    Get the gene and relationship in the pathway graph
    :param img_identifier_output_path: Path of the pathway graph
    :return: Get gene and relationship results
    '''
    for dir in os.listdir(img_identifier_output_path):
        if any((file.endswith('.jpg')) for file in os.listdir(os.path.join(img_identifier_output_path, dir,
                                                                           'img'))):
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
            args.dataset = os.path.join(img_identifier_output_path, dir)
            run_model(cfg, None, **vars(args))
        else:
            shutil.rmtree(os.path.join(img_identifier_output_path, dir))
            if os.path.exists(os.path.join(img_identifier_output_path, dir)):
                os.removedirs(os.path.join(img_identifier_output_path, dir))


def path_CLIP_inference():
    model_path = "model/my_best.pt"
    image_path = 'result'
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    gt_list = []
    pre_list = []
    for cate_dict in os.listdir(image_path):
        relation, _ = get_relation(cate_dict)
        gene_pari = cate_dict.split("+")
        query_list = [gene_pari[0] + ' activates ' + gene_pari[1], gene_pari[0] + ' inhibits ' + gene_pari[1],
                      gene_pari[1] + ' activates ' + gene_pari[0], gene_pari[1] + ' inhibits ' + gene_pari[0], ]
        model, text_embeddings = get_text_embeddings(query_list, model_path)
        for image in os.listdir(os.path.join(image_path, cate_dict)):
            gt, pre = find_matches1(model, os.path.join(image_path, cate_dict, image), text_embeddings, image, relation)
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

if __name__ == '__main__':
    pdf_path = 'paper'
    img_path = 'extract_img'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    identifier_model = 'model/train3/csv_retinanet_4.pt'
    img_identifier_output_path = 'result'
    if not os.path.exists(img_identifier_output_path):
        os.makedirs(img_identifier_output_path)
    # get_images(pdf_path, img_path, identifier_model, img_identifier_output_path)
    get_gene_relation(img_identifier_output_path)
    path_CLIP_inference()
