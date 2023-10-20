import json

# path = 'result/31161718_PBI-18-155/img/gene_name/31161718_PBI-18-155_page4_16_elements.json'
# path = "result/31161718_PBI-18-155/img/relation/31161718_PBI-18-155_page4_16_relation.json"
# path = "result/31161718_PBI-18-155/img/relation_subimage/31161718_PBI-18-155_page4_16.jpg"
# path = "result/31161718_PBI-18-155/img/relation/31161718_PBI-18-155_page4_16_object_detection.json"
path = "result/31161718_PBI-18-155/img/relation/31161718_PBI-18-155_page4_16_o_d_ele.json"

text = json.load(open(path))
file = open('31161718_PBI-18-155_page4_16_o_d_ele.json', 'w', encoding='utf-8')  # text to json file
file_content = json.dumps(text, indent=2, ensure_ascii=False)
file.write(file_content)