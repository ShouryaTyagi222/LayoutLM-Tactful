# python prepare_annots.py -i /data/circulars/DATA/LayoutLM/docvqa_dataset/Images -c /data/circulars/DATA/LayoutLM/docvqa_dataset/raw_data -q /data/circulars/DATA/layoutLM+Tactful
import argparse
import json
import os
from PIL import Image
from tqdm import tqdm

IMG_WIDTH=224
IMG_HEIGHT=244

def main(args):
    data_files = os.listdir(args.data_files_folder)

    final_data = []
    for data_file in data_files:
        with open(os.path.join(args.data_files_folder,data_file), 'r') as f:
            data = json.load(f)

        
        for da in tqdm(data):
            img_name=da["data"]['ocr'].split('/')[-1]
            annotations=da['annotations']
            data=[]
            for annotation in annotations:
                for annot in annotation['result']:
                    
                    if annot['from_name']=='bbox':
                        continue
                    if annot['type']=='rectanglelabels':
                        label=annot['value']['rectanglelabels'][0]
                        x=int(annot['value']['x']/100*IMG_WIDTH)
                        y=int(annot['value']['y']/100*IMG_HEIGHT)
                        width=int(annot['value']['width']/100*IMG_WIDTH)
                        height=int(annot['value']['height']/100*IMG_HEIGHT)
                        
                        if label=='Stamps/Seals':
                            label='Stamps-Seals'
                        
                        # img.save(os.path.join(args.query_path,label,img_name))
                        data.append({'x':x,'y':y,'w':width,'h':height,'label':label})
            final_data.append({'file_name':img_name, 'annotations':data})
    
    with open(os.path.join(args.output_dir,'full_data_annots.json'), 'w') as json_file:
        json.dump(final_data, json_file,indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Query Image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_dir", default=None, type=str, help="Path to the image Directory")
    parser.add_argument("-c", "--data_files_folder", default=None, type=str, help="Path to the raw json files folder")
    parser.add_argument("-q", "--output_dir", default='/', type=str, help="Path to the Output query image Folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)