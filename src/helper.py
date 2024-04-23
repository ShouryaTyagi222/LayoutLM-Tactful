import os
import shutil
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import Features, Sequence, Value, Array2D, Array3D
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
from detectron2.engine import DefaultPredictor

# Append parent directory to sys.path to import custom modules
import sys
sys.path.append("../")

# Import custom configurations
from configs import *

# Initialize RougeScorer with desired metrics
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def create_model(cfg):
    """
    Create and return a default predictor model using the provided configuration.

    Args:
        cfg (CfgNode): Detectron2 configuration for model prediction.

    Returns:
        DefaultPredictor: Instance of DefaultPredictor for making predictions.
    """
    tester = DefaultPredictor(cfg)
    return tester

def cal_rouge_scores(target_question, questions, o_answers, p_answers):
    """
    Calculate Rouge scores (precision, recall, F1) for generated answers compared to reference answers.

    Args:
        target_question (str): Target question for which to calculate specific RougeL score.
        questions (list): List of questions corresponding to answers.
        o_answers (list): List of original (reference) answers.
        p_answers (list): List of predicted (generated) answers.

    Returns:
        Tuple: Tuple containing Rouge scores and specific target RougeL score.
               (rouge1_p, rouge1_r, rouge1_f, rouge2_p, rouge2_r, rouge2_f,
                rougel_p, rougel_r, rougel_f, target_rougel_f)
    """
    # Initialize lists to store Rouge scores
    rouge1_p = []
    rouge1_r = []
    rouge1_f = []
    rouge2_p = []
    rouge2_r = []
    rouge2_f = []
    rougel_p = []
    rougel_r = []
    rougel_f = []
    target_rougel_f = []

    # Calculate Rouge scores for each question-answer pair
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

        # Record RougeL fmeasure for the target question
        if question == target_question:
            target_rougel_f.append(scores['rougeL'].fmeasure)
    
    # Calculate mean Rouge scores
    rouge1_p = np.mean(rouge1_p)
    rouge1_r = np.mean(rouge1_r)
    rouge1_f = np.mean(rouge1_f)
    rouge2_p = np.mean(rouge2_p)
    rouge2_r = np.mean(rouge2_r)
    rouge2_f = np.mean(rouge2_f)
    rougel_p = np.mean(rougel_p)
    rougel_r = np.mean(rougel_r)
    rougel_f = np.mean(rougel_f)
    target_rougel_f = np.mean(target_rougel_f)

    return (
        rouge1_p, rouge1_r, rouge1_f,
        rouge2_p, rouge2_r, rouge2_f,
        rougel_p, rougel_r, rougel_f,
        target_rougel_f
    )

def get_output(batch, model, model_type, device):
    """
    Forward pass through a model with input batch data based on the model type.

    Args:
        batch (dict): Input batch data containing input_ids, bbox, image, start_positions, and end_positions.
        model (torch.nn.Module): Model instance to perform inference.
        model_type (str): Type of the model ('layoutlmv2' or 'layoutlmv3').
        device (torch.device): Device (CPU or GPU) on which the model should run.

    Returns:
        torch.Tensor: Output predictions from the model.
    """
    if model_type.lower() == 'layoutlmv2':
        input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
        bbox = batch["bbox"].to(device=device, dtype=torch.long)
        image = batch["image"].to(device=device, dtype=torch.float32)
        start_positions = batch["start_positions"].to(device=device, dtype=torch.long)
        end_positions = batch["end_positions"].to(device=device, dtype=torch.long)

        # Forward pass through LayoutLMv2 model
        outputs = model(input_ids=input_ids, bbox=bbox, image=image,
                        start_positions=start_positions, end_positions=end_positions)

    elif model_type.lower() == 'layoutlmv3':
        input_ids = batch["input_ids"].to(device=device)
        bbox = batch["bbox"].to(device=device)
        image = batch["image"].to(device=device, dtype=torch.float)
        start_positions = batch["start_positions"].to(device=device)
        end_positions = batch["end_positions"].to(device=device)

        # Forward pass through LayoutLMv3 model
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=image,
                        start_positions=start_positions, end_positions=end_positions)

    else:
        print('ENTER THE CORRECT NAME OF THE MODEL !!')
        outputs = None

    return outputs


def write_predictions_to_txt(questions, original_answers, predicted_answers, output_file):
    """
    Write question, original answer, and predicted answer to a text file.

    Args:
        questions (list): List of questions corresponding to the answers.
        original_answers (list): List of original (reference) answers.
        predicted_answers (list): List of predicted (generated) answers.
        output_file (str): Path to the output text file.
    """
    with open(output_file, 'a') as f:
        for question, original_answer, predicted_answer in zip(questions, original_answers, predicted_answers):
            f.write(f"Question: {question}\n")
            f.write(f"Original Answer: {original_answer}\n")
            f.write(f"Predicted Answer: {predicted_answer}\n")
            f.write("\n")


def convert_to_custom_format(original_dataset, image_dir, banned_files):
    """
    Convert the original dataset into a custom format suitable for model input.

    Args:
        original_dataset (list): List of dictionaries representing the original dataset.
        image_dir (str): Path to the directory containing images.
        banned_files (list): List of filenames to be excluded from the custom dataset.

    Returns:
        list: Custom formatted dataset containing processed examples.
    """
    custom_dataset = []

    for document in original_dataset:
        file_name = document["file_name"]
        if file_name in banned_files:
            continue

        image = cv2.imread(os.path.join(image_dir, file_name))
        image = cv2.resize(image, (224, 224))

        for qa_pair in document["q_and_a"]:
            question = qa_pair.get('question', '')
            input_ids = np.array(qa_pair.get("input_ids", []))
            input_ids = input_ids[:512]  # Limit input_ids to first 512 tokens

            start_positions = qa_pair.get("start_idx", -1)
            end_positions = qa_pair.get("end_idx", -1)

            if start_positions > 512 or end_positions > 512:
                continue

            custom_example = {
                'question': question,
                'input_ids': input_ids,
                'bbox': np.pad(np.array(document["boxes"]), ((0, 512 - len(document["boxes"])), (0, 0)), mode='constant', constant_values=0),
                'image': torch.tensor(image).permute(2, 0, 1),
                'start_positions': start_positions,
                'end_positions': end_positions,
            }

            custom_dataset.append(custom_example)

    return custom_dataset


def custom_collate(batch):
    """
    Custom collate function to handle different data types in a batch.

    Args:
        batch (list): List of batched samples.

    Returns:
        dict or torch.Tensor: Collated batch of data.
    """
    elem_type = type(batch[0])

    if elem_type in (int, float):
        return torch.tensor(batch)
    elif elem_type is torch.Tensor:
        return torch.stack(batch, dim=0)
    elif elem_type is list:
        return [custom_collate(samples) for samples in zip(*batch)]
    elif elem_type is dict:
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return default_collate(batch)

def load_data(input_file, img_dir, batch_size, banned_txt_path):
    """
    Load data from input JSON file and prepare a DataLoader for training or evaluation.

    Args:
        input_file (str): Path to the input JSON file containing dataset information.
        img_dir (str): Path to the directory containing images.
        batch_size (int): Batch size for DataLoader.
        banned_txt_path (str): Path to the text file containing banned filenames.

    Returns:
        DataLoader: DataLoader instance for the custom dataset.
    """
    with open(banned_txt_path) as f:
        banned_files = [x.strip() for x in f.readlines()]

    encoded_dataset = json.load(open(input_file))
    encoded_dataset = convert_to_custom_format(encoded_dataset, img_dir, banned_files)

    dataloader = DataLoader(encoded_dataset, batch_size=batch_size, collate_fn=custom_collate)
    return dataloader

def crop_object(image, box, ground_truth=False):
    """
    Crop an object from an image based on the provided box coordinates.

    Args:
        image (PIL.Image): Input image to crop.
        box (list): List of box coordinates [x_top_left, y_top_left, x_bottom_right, y_bottom_right].
        ground_truth (bool): Whether the box coordinates represent ground truth (default: False).

    Returns:
        PIL.Image: Cropped image of the object.
    """
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = box
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2

    try:
        crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
    except Exception as e:
        crop_img = None  # Handle exception gracefully

    return crop_img

def crop_images_classwise_ground_truth(train_json_path, train_images_path, src_path, dest_path, category):
    """
    Crop images based on ground truth annotations for a specific category.

    Args:
        train_json_path (str): Path to the JSON file containing training annotations.
        train_images_path (str): Path to the directory containing training images.
        src_path (str): Source path where images are stored.
        dest_path (str): Destination path to save cropped images.
        category (str): Category of objects to crop.

    Returns:
        None
    """
    category = category.lower()
    obj_im_dir = os.path.join(dest_path, 'obj_images')
    os.makedirs(obj_im_dir, exist_ok=True)
    no_of_objects = 0

    with open(train_json_path) as f:
        data = json.load(f)

    train_images = os.listdir(train_images_path)
    file_names = os.listdir(src_path)
    
    for annot in tqdm(data):
        img_name = annot['file_name']
        if img_name in file_names and img_name in train_images:
            img_annots = annot['annotations']
            img = cv2.imread(os.path.join(src_path, img_name))
            img = cv2.resize(img, (224, 224))
            
            for i, img_annot in enumerate(img_annots):
                if img_annot['label'].lower() == category:
                    no_of_objects += 1
                    x, y, w, h = int(img_annot['x']), int(img_annot['y']), int(img_annot['w']), int(img_annot['h'])
                    crp_img = img[y:y+h, x:x+w]
                    
                    if y+h < 224 and x+w < 224:
                        cv2.imwrite(os.path.join(obj_im_dir, category, f'{img_name}_{i}.png'), crp_img)

def crop_images_classwise(src_path, dest_path, proposal_budget):
    """
    Crop images based on detected object proposals from a trained object detection model.

    Args:
        src_path (str): Source path where images are stored.
        dest_path (str): Destination path to save cropped images.
        proposal_budget (int): Maximum number of object proposals to consider per image.

    Returns:
        None
    """
    obj_im_dir = os.path.join(dest_path, 'obj_images')
    os.makedirs(obj_im_dir, exist_ok=True)
    model = create_model(cfg)
    no_of_objects = 0

    for d in tqdm(os.listdir(src_path)):
        image = cv2.imread(os.path.join(src_path, d))
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        
        images = model.model.preprocess_image(inputs)
        features = model.model.backbone(images.tensor)
        proposals, _ = model.model.proposal_generator(images, features)
        instances, _ = model.model.roi_heads(images, features, proposals)
        
        boxes = instances[0].pred_boxes
        classes = instances[0].pred_classes.cpu().numpy().tolist()
        max_score_order = torch.argsort(instances[0].scores).tolist()

        if proposal_budget > len(max_score_order):
            proposal_budget = len(max_score_order)

        img = Image.open(os.path.join(src_path, d))
        for idx, box in enumerate(boxes[max_score_order[:proposal_budget]]):
            no_of_objects += 1
            box = box.detach().cpu().numpy()
            crop_img = crop_object(img, box)
            
            try:
                crop_img.save(os.path.join(obj_im_dir, R_MAPPING[str(classes[idx])], f'{d.replace(".png", "")}_{idx}.png'))
            except Exception as e:
                print(e)

    print(f"Number of objects: {no_of_objects}")

def Random_wrapper(image_list, budget=10):
    """
    Randomly selects a subset of images from the given list.

    Args:
        image_list (list): List of image filenames.
        budget (int): Number of images to select randomly (default: 10).

    Returns:
        list: List of randomly selected image filenames.
    """
    rand_idx = np.random.permutation(len(image_list))[:budget]
    rand_idx = rand_idx.tolist()
    Random_results = [image_list[i] for i in rand_idx]
    return Random_results

def change_dir(image_results, src_dir, dest_dir):
    """
    Moves selected images from source directory to destination directory.

    Args:
        image_results (list): List of selected image filenames.
        src_dir (str): Source directory path.
        dest_dir (str): Destination directory path.

    Returns:
        None
    """
    for image in image_results:
        source_img = os.path.join(src_dir, image)
        destination_img = os.path.join(dest_dir, os.path.basename(image))

        # Create destination directories if they don't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        try:
            # Copy image from source to destination
            shutil.copy(source_img, destination_img)
            # Remove image from source directory
            os.remove(source_img)
        except Exception as e:
            print(f"Error occurred while moving file: {e}")

def remove_dir(dir_name):
    """
    Removes a directory and its contents recursively.

    Args:
        dir_name (str): Directory path to be removed.

    Returns:
        None
    """
    try:
        shutil.rmtree(dir_name)
    except Exception as e:
        print(f"Error occurred while removing directory: {e}")

def create_dir(dir_name):
    """
    Creates a directory if it doesn't exist.

    Args:
        dir_name (str): Directory path to be created.

    Returns:
        None
    """
    try:
        os.makedirs(dir_name, exist_ok=True)
    except Exception as e:
        print(f"Error occurred while creating directory: {e}")

def get_original_images_path(subset_result, img_dir):
    """
    Extracts original image filenames from a subset of results.

    Args:
        subset_result (list): List of image filenames.
        img_dir (str): Directory path where images are located.

    Returns:
        list: List of original image filenames without suffixes.
    """
    return ["_".join(os.path.basename(x).split("_")[:-1]) + '.png' for x in subset_result]

def aug_train_subset(subset_result, train_data_json, lake_data_json, budget, src_dir, dest_dir):
    """
    Augments training dataset by moving selected images from lake dataset to train dataset.

    Args:
        subset_result (list): List of selected image filenames.
        train_data_json (str): Path to the training dataset JSON file.
        lake_data_json (str): Path to the lake dataset JSON file.
        budget (int): Number of images selected for augmentation.
        src_dir (str): Source directory where images are located.
        dest_dir (str): Destination directory where images will be moved.

    Returns:
        None
    """
    with open(lake_data_json, 'r') as f:
        lake_dataset = json.load(f)

    with open(train_data_json, 'r') as f:
        train_dataset = json.load(f)

    new_lake_data = []

    for data in lake_dataset:
        if data['file_name'] in subset_result:
            train_dataset.append(data)
        else:
            new_lake_data.append(data)

    # Move selected images from lake dataset to train dataset
    change_dir(subset_result, src_dir, dest_dir)
    print('\n SHIFT TRAIN LEN :', len(train_dataset))

    # Update lake dataset and train dataset JSON files
    with open(lake_data_json, 'w') as f:
        json.dump(new_lake_data, f, indent=4)

    with open(train_data_json, 'w') as f:
        json.dump(train_dataset, f, indent=4)