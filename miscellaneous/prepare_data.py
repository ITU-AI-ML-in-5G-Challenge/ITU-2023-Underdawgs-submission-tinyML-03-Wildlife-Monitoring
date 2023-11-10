import json
import cv2
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
annot_dir = "./pose_estimation/annotation_coco/"

id_final = []
id_final2 = []
image_path_final = []
animal_class_final = []
train_test_final = []
x_final = []
y_final = []
w_final = []
h_final = []
width_final = []
height_final = []
df = pd.DataFrame()

for Proto in tqdm(os.listdir(annot_dir)):
    if Proto.startswith("ak_P3") and "fish" not in Proto and "amphi" not in Proto:
        # print(Proto)
        for sub_proto in os.listdir(annot_dir+Proto):
            file_path = annot_dir+Proto+'/'+sub_proto
            # print(file_path)
            file_ = open(file_path)
            full_data = json.load(file_) 
            # print(file_path)
            try: 
                assert len(full_data['images'])==len(full_data['annotations'])

                for indi_data in full_data['images']:
                    id_final.append(indi_data["id"])
                    image_path_final.append(indi_data['file_name'])
                    height_final.append(indi_data['height'])
                    width_final.append(indi_data['width'])

                for indi_data in full_data["annotations"]:
                    id_final2.append(indi_data['id'])
                    animal_class_final.append(indi_data["animal"])
                    train_test_final.append(indi_data["train_test"])
                    x_final.append(indi_data["bbox"][0])
                    y_final.append(indi_data["bbox"][1])
                    w_final.append(indi_data["bbox"][2])
                    h_final.append(indi_data["bbox"][3])
   
                 
            except:
                print(file_path)
color = (0, 255, 0)  # Green color in BGR format (change as needed)
thickness = 2  # Line thickness (change as needed)

    

df['id_final']=id_final
df['filename']=image_path_final
df['class']=animal_class_final
df['train_test_final']=train_test_final
df['xmin']=x_final
df['ymin']=y_final
df['xmax']=np.add(x_final,w_final)
df['ymax']=np.add(y_final,h_final)
df['height']=height_final
df['width']=width_final

df.to_csv('df.csv', index=False)
# for i in df.iterrows():
#     i=i[1]
#     # print(i[["xmin_final","ymin_final","xmax_final","ymax_final"]].values)
#     xmin_final,ymin_final,xmax_final,ymax_final=i[["xmin_final","ymin_final","xmax_final","ymax_final"]].values
#     print(i["image_path_final"])
#     img = cv2.imread("pose_estimation/dataset-001/dataset/"+i["image_path_final"])
#     # new_width = i["width_final"]*0.6674772036000001
#     # new_height = i["height_final"]*0.6674772036000001
#     # img = cv2.resize(
#     #     img, (int(new_width), int(new_height)), interpolation=cv2.INTER_LINEAR
#     # )
#     cv2.imwrite(i["image_path_final"].split('/')[1],cv2.rectangle(img, (int(xmin_final), int(ymin_final)), (int(xmin_final)+int(xmax_final), int(ymin_final)+int(ymax_final)), color, thickness))
    # break

train = df[df['train_test_final']=='train']
train.to_csv("train.csv", index=False)
test = df[df['train_test_final']=='test']
train.to_csv("test.csv", index=False)
