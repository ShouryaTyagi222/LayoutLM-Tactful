import os, shutil, json ,cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import Features, Sequence, Value, Array2D, Array3D
from torch.utils.data.dataloader import default_collate
import sys
sys.path.append("../")
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

from detectron2.engine import DefaultPredictor

import sys
sys.path.append("../")

from configs import *


def create_model(cfg):
    tester = DefaultPredictor(cfg)
    return tester
    

def cal_rouge_scores(target_question, questions, o_answers, p_answers):
    rouge1_p=[]
    rouge1_r=[]
    rouge1_f=[]
    rouge2_p=[]
    rouge2_r=[]
    rouge2_f=[]
    rougel_p=[]
    rougel_r=[]
    rougel_f=[]
    target_rougel_f=[]
    print(target_question)

    for question, hypothesis, reference in zip(questions, o_answers, p_answers):
        scores = scorer.score(hypothesis, reference)
        rouge1_p.append(scores['rouge1'].precision)
        rouge1_r.append(scores['rouge1'].recall)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_p.append(scores['rouge2'].precision)
        rouge2_r.append(scores['rouge2'].recall)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougel_p.append(scores['rougeL'].precision)
        rougel_r.append(scores['rougeL'].recall)
        rougel_f.append(scores['rougeL'].fmeasure)
        if question == target_question:
            target_rougel_f.append(scores['rougeL'].fmeasure)
    
    rouge1_p=np.mean(rouge1_p)
    rouge1_r=np.mean(rouge1_r)
    rouge1_f=np.mean(rouge1_f)
    rouge2_p=np.mean(rouge2_p)
    rouge2_r=np.mean(rouge2_r)
    rouge2_f=np.mean(rouge2_f)
    rougel_p=np.mean(rougel_p)
    rougel_r=np.mean(rougel_r)
    rougel_f=np.mean(rougel_f)
    target_rougel_f=np.mean(target_rougel_f)

    return rouge1_p, rouge1_r, rouge1_f, rouge2_p, rouge2_r, rouge2_f, rougel_p, rougel_r, rougel_f, target_rougel_f

def get_output(batch,model,model_type,device):
    if model_type.lower()=='layoutlmv2':
        input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
        bbox = batch["bbox"].to(device=device, dtype=torch.long)  # No need to specify data type
        image = batch["image"].to(device=device, dtype=torch.float32)  # Assuming image data type is float32
        start_positions = batch["start_positions"].to(device=device, dtype=torch.long)  # No need to specify data type
        end_positions = batch["end_positions"].to(device=device, dtype=torch.long)  # No need to specify data type

        # forward + backward + optimize
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, start_positions=start_positions, end_positions=end_positions)

    elif model_type.lower()=='layoutlmv3':
        # get the inputs;
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        image = batch["image"].to(device, dtype=torch.float)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        # forward + backward + optimize
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=image, start_positions=start_positions, end_positions=end_positions)

    else:
        print('ENTER THE CORRECT NAME OF THE MODEL !!')
    
    return outputs
    
def write_predictions_to_txt(questions, original_answers, predicted_answers, output_file):
    with open(output_file, 'a') as f:
        for question, original_answer, predicted_answer in zip(questions, original_answers, predicted_answers):
            f.write(f"Question: {question}\n")
            f.write(f"Original Answer: {original_answer}\n")
            f.write(f"Predicted Answer: {predicted_answer}\n")
            f.write("\n")

def convert_to_custom_format(original_dataset,image_dir,banned_files):
    custom_dataset = []

    count = 0

    for doc_id, document in enumerate(original_dataset):
        # File Name
        file_name = document["file_name"]

        # Load The File
        image = cv2.imread(f'{image_dir}/{file_name}')
        image=cv2.resize(image,(224,224))

        # Skip if the file is in the banned files
        # Skip if the file is in the banned files
        if file_name in banned_files:
            continue

        try:
            for qa_id, qa_pair in enumerate(document["q_and_a"]):
                question = qa_pair.get('question',-1)
                boxes_arr = np.array(document["boxes"])
                # Pad the boxes array to 512
                padded_boxes = np.pad(boxes_arr, ((0, 512 - len(boxes_arr)), (0, 0)), mode='constant', constant_values=0)
                # # Get the Channels first
                bbox = boxes_arr  # Placeholder for bbox
                image_tensor = torch.tensor(image).clone().detach()
                image_tensor = image_tensor.permute(2, 0, 1)

                # Fill in your data processing logic here to populate input_ids, bbox, attention_mask, token_type_ids, and image
                input_ids = np.array(qa_pair.get("input_ids", -1))
                # Just take the first 512 tokens
                input_ids = input_ids[:512]

                # Fill in your data processing logic here to populate input_ids, bbox, attention_mask, token_type_ids, and image

                start_positions = qa_pair.get("start_idx", -1)
                end_positions = qa_pair.get("end_idx", -1)

                if start_positions > 512:
                    start_positions = -1
                    continue

                if end_positions > 512:
                    end_positions = -1
                    continue

                custom_example = {
                    'question': question,
                    'input_ids': input_ids,
                    'bbox': padded_boxes,
                    'image': image_tensor,
                    'start_positions': start_positions,
                    'end_positions': end_positions,
                }

                custom_dataset.append(custom_example)
                count += 1
        except Exception as e:
            print(f"Error processing Document {doc_id}, QA {qa_id}: {str(e)}")
            count += 1
            continue

    features = {
        'question': Value(dtype='string'),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'start_positions': Value(dtype='int64'),
        'end_positions': Value(dtype='int64'),
    }
    return custom_dataset

def custom_collate(batch):
    elem_type = type(batch[0])

    if elem_type in (int, float):
        return torch.tensor(batch)
    elif elem_type is torch.Tensor:
        return torch.stack(batch, dim=0)
    elif elem_type is list:
        # Handle lists differently, especially sequences
        return [custom_collate(samples) for samples in zip(*batch)]
    elif elem_type is dict:
        # Handle dictionaries
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        # For other types, use the default_collate behavior
        return default_collate(batch)

def load_data(input_file, img_dir, batch_size, banned_txt_path):
    with open(banned_txt_path) as f:
        banned_files = f.readlines()

    banned_files = [x.strip() for x in banned_files]

    print('Loading the Data')

    encoded_dataset = json.load(open(input_file))
    encoded_dataset = convert_to_custom_format(encoded_dataset, img_dir, banned_files)

    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size, collate_fn=custom_collate)
    return dataloader

def crop_object(image, box, ground_truth=False):
    """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
    """
    if (not ground_truth):
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
    else:
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[0] + box[2]
        y_bottom_right = box[1] + box[3]
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2

    try:
        crop_img = image.crop((int(x_top_left), int(y_top_left),
                               int(x_bottom_right), int(y_bottom_right)))
    except Exception as e:
        pass

    return crop_img

'''
Returns the list of cropped images based on the objects. The method make use of ground truth to crop the image.
'''
def crop_images_classwise_ground_truth(train_json_path, src_path, dest_path,
                                       category: str):
    category=category.lower()
    if not os.path.exists(dest_path + '/obj_images'):
        os.makedirs(dest_path + '/obj_images')
    obj_im_dir = dest_path + '/obj_images'
    no_of_objects = 0
    with open(train_json_path) as f:
        data = json.load(f)
    file_names = os.listdir(src_path)
    
    for annot in tqdm(data):
        img_name = annot['file_name']
        # print(img_name)
        if img_name in file_names:
            img_annots=annot['annotations']
            img=cv2.imread(os.path.join(src_path,img_name))
            img=cv2.resize(img,(224,224))
            # print(img_name,':',img_annots)
            for i,img_annot in enumerate(img_annots):
                if img_annot['label'].lower()==category.lower():
                    no_of_objects += 1
                    x,y,w,h=int(img_annot['x']),int(img_annot['y']),int(img_annot['w']),int(img_annot['h'])
                    crp_img=img[y:y+h,x:x+w]
                    if y+h<224 and x+w<224:
                        cv2.imwrite(os.path.join(obj_im_dir,category,'.'.join(img_name.split('.')[:-1])+'_'+str(i)+'.png'),crp_img)
                        

'''
Returns the list of cropped image based on the objects. The method uses the trained object detection\
     model to get bouding box and crop the images.
'''
# def crop_images_classwise(lake_json_path, src_path, dest_path,
#                           proposal_budget: int):
#     with open(lake_json_path) as f:
#         data = json.load(f)
#     if not os.path.exists(dest_path + '/obj_images'):
#         os.makedirs(dest_path + '/obj_images')
#     obj_im_dir = dest_path + '/obj_images'
#     no_of_objects = 0
#     file_names = os.listdir(src_path)
#     # print('source path',src_path)
#     # print('ground truth files name', file_names)
#     for annot in tqdm(data):
#         img_name = annot['file_name']
#         if img_name in file_names:
#             img_annots=annot['annotations']
#             # print('Image :',os.path.join(src_path,img_name))
#             img=cv2.imread(os.path.join(src_path,img_name))
#             img=cv2.resize(img,(224,224))
#             for i,img_annot in enumerate(img_annots):
#                 category = img_annot['label'].lower()
#                 if not os.path.exists(os.path.join(obj_im_dir, category)):
#                     os.mkdir(os.path.join(obj_im_dir, category))
#                 no_of_objects += 1
#                 x,y,w,h=int(img_annot['x']),int(img_annot['y']),int(img_annot['w']),int(img_annot['h'])
#                 crp_img=img[y:y+h,x:x+w]
#                 # print(os.path.join(obj_im_dir,category,img_name.split('.')[0]+'_'+str(i)+'.png'),crp_img)
#                 try:
#                     cv2.imwrite(os.path.join(obj_im_dir,category,'.'.join(img_name.split('.')[:-1])+'_'+str(i)+'.png'),crp_img)
#                 except:
#                     print('crp img',img_name)

#     print("Number of objects: " + str(no_of_objects))

def crop_images_classwise(src_path, dest_path,
                          proposal_budget: int):
    if not os.path.exists(dest_path + '/obj_images'):
        os.makedirs(dest_path + '/obj_images')
    model = create_model(cfg)
    obj_im_dir = dest_path + '/obj_images'
    no_of_objects = 0
    print(src_path)
    for d in tqdm(os.listdir(src_path)):
        image = cv2.imread(os.path.join(src_path, d))
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        images = model.model.preprocess_image(inputs)

        features = model.model.backbone(images.tensor)
        proposals, _ = model.model.proposal_generator(images, features)
        instances, _ = model.model.roi_heads(images, features,
                                                     proposals)
        boxes = instances[0].pred_boxes
        classes = instances[0].pred_classes.cpu().numpy().tolist()
        max_score_order = torch.argsort(instances[0].scores).tolist()

        if (proposal_budget > len(max_score_order)):
            proposal_budget = len(max_score_order)

        for singleclass in classes:
            if not os.path.exists(
                    os.path.join(dest_path, 'obj_images',
                                 R_MAPPING[str(singleclass)])):
                os.makedirs(
                    os.path.join(dest_path, 'obj_images',
                                 R_MAPPING[str(singleclass)]))

        img = Image.open(os.path.join(src_path, d))
        for idx, box in enumerate(
                list(boxes[max_score_order[:proposal_budget]])):
            no_of_objects += 1
            box = box.detach().cpu().numpy()

            crop_img = crop_object(img, box)
            try:
                crop_img.save(
                    os.path.join(
                        obj_im_dir, R_MAPPING[str(classes[idx])],
                        d.replace(".png", "") + "_" + str(idx) + ".png"))
            except Exception as e:
                print(e)

    print("Number of objects: " + str(no_of_objects))

def Random_wrapper(image_list, budget=10):
    rand_idx = np.random.permutation(len(image_list))[:budget]
    rand_idx = rand_idx.tolist()
    Random_results = [image_list[i] for i in rand_idx]

    return Random_results

def change_dir(image_results, src_dir, dest_dir):
    for image in image_results:
        source_img = os.path.join(src_dir[0],image)
        destination_img = os.path.join(dest_dir[0], os.path.basename(image))
        if not os.path.exists(dest_dir[0]) or not os.path.exists(dest_dir[1]):
            os.mkdir(dest_dir[0])
            os.mkdir(dest_dir[1])

        try:
            shutil.copy(source_img, destination_img)
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")

        # For other errors
        except Exception as e:
            print("Error occurred while copying file.", e)


        # removing the data from the lake data
        try:
            os.remove(source_img)
        except:
            pass

def remove_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except:
        pass

def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass

def get_original_images_path(subset_result:list,img_dir:str):
    return ["_".join(os.path.basename(x).split("_")[:-1])+'.png' for x in subset_result]

def aug_train_subset(subset_result, train_data_json, lake_data_json, budget, src_dir, dest_dir):
    with open(lake_data_json, mode="r") as f:
        lake_dataset = json.load(f)
    with open(train_data_json, mode="r") as f:
        train_dataset = json.load(f)

    new_lake_data=[]

    for data in lake_dataset:
        if data['file_name'] in subset_result:
            train_dataset.append(data)
        else:
            new_lake_data.append(data)

    #moving data from lake set to train set.
    change_dir(subset_result, src_dir, dest_dir)
    print('\n SHIFT TRAIN LEN :',len(train_dataset))

    #changing the file for annotations
    with open(lake_data_json, mode='w') as f:
        json.dump(new_lake_data,f,indent=4)
    with open(train_data_json,'w') as f:
        json.dump(train_dataset,f,indent=4)