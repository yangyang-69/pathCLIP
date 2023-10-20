#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pure_pathway_1 
@File    ：copy_already_result.py
@Author  ：yang
@Date    ：2023/3/16 19:47 
'''
import os
import shutil


def check_already_page(result_file):
    if not os.path.exists("already_result"):
        os.makedirs("already_result")
    for pmid in os.listdir(result_file):
        image_list = []
        for file in os.listdir(os.path.join(result_file,pmid,'img')):
            if file.find(".jpg"):
                image_list.append(file)
        if len(image_list) > 0:
            file_list = []
            print(pmid)
            for file in os.listdir(os.path.join(result_file, pmid, 'img')):
                if file == "gene_name":
                    file_list.append(file)
                if file == "relation":
                    file_list.append(file)
                if file == "relation_subimage":
                    file_list.append(file)
            if len(file_list) == 3:
                print("copy")
                shutil.copytree(os.path.join(result_file, pmid), os.path.join("already_result",pmid))
                print("delete")
                shutil.rmtree(os.path.join(result_file, pmid))

if __name__ == '__main__':
    result_file = "result"
    check_already_page(result_file)