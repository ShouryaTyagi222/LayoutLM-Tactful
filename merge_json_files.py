import os
import json

input_folder='/data/circulars/DATA/LayoutLM/docvqa_dataset/processed_data_v3'
output_folder='/data/circulars/DATA/LayoutLM/docvqa_dataset'
output_file_name_V2 = 'full_data_v2_3.json'
output_file_name_V3 = 'full_data_v3_3.json'


input_files = os.listdir(input_folder)
final_data_V2 = []
final_data_V3 = []

for input_file in input_files:
    if input_file.split('.')[0].split('_')[-1]=='v2':
        data = json.load(open(os.path.join(input_folder,input_file)))
        final_data_V2.extend(data)
    elif input_file.split('.')[0].split('_')[-1]=='v3':
        data = json.load(open(os.path.join(input_folder,input_file)))
        final_data_V3.extend(data)

print('output file length :',len(final_data_V2))

with open(output_folder+'/'+output_file_name_V2, "w") as f:
    json.dump(final_data_V2, f, indent=4)
with open(output_folder+'/'+output_file_name_V3, "w") as f:
    json.dump(final_data_V3, f, indent=4)