# pip install sahi
# python split_data.py -i '/data/circulars/DATA/LayoutLM/docvqa_dataset/full_data_v3_3.json' -d '/data/circulars/DATA/LayoutLM/docvqa_dataset/Images' -o '/data/circulars/DATA/layoutLM+Tactful/layoutlmv3_data'
import random
import os
import json
import argparse
import warnings
import shutil
warnings.filterwarnings("ignore")
from PIL import Image


def main(args):
    data_file=args.data_file
    image_dir=args.img_dir
    output_dir=args.output_dir
    train_split = int(args.train_length)
    val_split = int(args.val_length)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    la=0
    va=0
    tr=0

    for folder in ['train','lake','val']:
        if not os.path.exists(os.path.join(output_dir,folder)):
            os.mkdir(os.path.join(output_dir,folder))

    with open(data_file, 'r') as f:
        data = json.load(f)

    train_data=[]
    val_data=[]
    lake_data=[]

    for document in data:
        img_name=document['file_name']
        img_path=os.path.join(image_dir,img_name)
        img=Image.open(img_path)
        
        split=random.randint(0,2)

        if tr<train_split and split==0:
            train_data.append(document)
            tr+=1
            img.save(os.path.join(output_dir,'train',img_name))
        elif va<val_split and split==1:
            val_data.append(document)
            va+=1
            img.save(os.path.join(output_dir,'val',img_name))
        else:
            lake_data.append(document)
            la+=1
            img.save(os.path.join(output_dir,'lake',img_name))

        print(img_name)
        print([tr,va,la])
        print([len(os.listdir(output_dir+'/train')),len(os.listdir(output_dir+'/val')),len(os.listdir(output_dir+'/lake'))])

    with open(os.path.join(output_dir,'docvqa_train.json'), 'w') as json_file:
        json.dump(train_data, json_file,indent=4)
    with open(os.path.join(output_dir,'docvqa_val.json'), 'w') as json_file:
        json.dump(val_data, json_file,indent=4)
    with open(os.path.join(output_dir,'docvqa_lake.json'), 'w') as json_file:
        json.dump(lake_data, json_file,indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Udop", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--data_file", default=None, type=str, help="Path to the input json file")
    parser.add_argument("-d", "--img_dir", default=None, type=str, help="Path to the image Directory")
    parser.add_argument("-o", "--output_dir", default='/', type=str, help="Path to the Output Folder")
    parser.add_argument("-t", "--train_length", default='10', type=str, help="Train split of the data")
    parser.add_argument("-v", "--val_length", default='200', type=str, help="Val split of the data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)