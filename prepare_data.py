from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# tokenizer Checkpoint
model_checkpoint_v2 = "microsoft/layoutlmv2-base-uncased"
model_checkpoint_v3 = "microsoft/layoutlmv3-base"

# path Inputs
input_files_folder = '/data/circulars/DATA/LayoutLM/docvqa_dataset/raw_data'
img_dir = '/data/circulars/DATA/LayoutLM/docvqa_dataset/Images'
output_dir='/data/circulars/DATA/LayoutLM/temp_dataset'
output_file_name_V2 = 'full_data_v2_3.json'
output_file_name_V3 = 'full_data_v3_3.json'

# Image width and height
image_width=224
image_height=224

# label to question mapping
label_to_question = {
    "Date Block" : "What is the date when the circular was issued?",
    "Subject Block" : "What is the subject of the circular?",
    "Adressed To" : "To whom is the circular addressed to?",
    "Addressed To Block": "What is the complete address of the recipient?",
    "Address of Issuing Authority": "What is the address of the issuing authority?",
    "Address Block" : "What is the complete address of the circular?",
    "Header Block" : "What is the header of the circular?",
    "Signature Block" : "Who is the signatory of the circular?",
    # "Body Block" : "What is the body of the circular?",
    "Copy-Forwarded To Block" : "Who is the circular copied to?",
    "Reference Id" : "What is the reference id of the circular?",
    "Reference Block": "What is the reference mentioned in the circular?",
    "Circular ID": "What is the identification number of the circular?",
}

model = ocr_predictor(pretrained=True)

def extract_rect_label_boxes(dataset, img_width, img_height):
    rect_label_boxes = []

    for annotation in dataset["annotations"]:
        if annotation["from_name"] == "rect-label":
            # Get the bounding box coordinates
            # x = annotation["value"]["x"] * img_width / 100
            # y = annotation["value"]["y"] * img_height / 100
            # width = annotation["value"]["width"] * img_width / 100
            # height = annotation["value"]["height"] * img_height / 100
            x = annotation["value"]["x"] * image_width / 100
            y = annotation["value"]["y"] * image_height / 100
            width = annotation["value"]["width"] * image_width / 100
            height = annotation["value"]["height"] * image_height / 100

            rect_label_boxes.append([x, y, width, height, annotation["value"]["rectanglelabels"][0]])

    return rect_label_boxes

def extract_ocr_boxes(data, img_width, img_height):
    ocr_boxes = []

    d_new = data["annotations"]

    for data in d_new:
        for entry in data['prediction']['result']:
            if entry["from_name"] == "transcription":
                # Convert normalized coordinates to pixel values
                x = entry["value"]["x"] * img_width / 100
                y = entry["value"]["y"] * img_height / 100
                width = entry["value"]["width"] * img_width / 100
                height = entry["value"]["height"] * img_height / 100

                ocr_boxes.append([x, y, width, height, entry["value"]["text"]])

    return ocr_boxes

def is_box_inside_2(rect_label_box, ocr_box, threshold):
    """
    Check if an OCR box is inside a rectangular label box with a given threshold.

    Parameters:
        rect_label_box (list): Coordinates of the rectangular label box in the format [x1, y1, x2, y2].
        ocr_box (list): Coordinates of the OCR box in the format [x1, y1, x2, y2].
        threshold (int): Threshold percentage for overlap.

    Returns:
        bool: True if the OCR box is inside the rectangular label box within the threshold, False otherwise.
    """

    # Calculate the area of the intersection
    x_overlap = max(0, min(rect_label_box[2], ocr_box[2]) - max(rect_label_box[0], ocr_box[0]))
    y_overlap = max(0, min(rect_label_box[3], ocr_box[3]) - max(rect_label_box[1], ocr_box[1]))
    intersection_area = x_overlap * y_overlap

    # Calculate the area of the OCR box
    ocr_area = (ocr_box[2] - ocr_box[0]) * (ocr_box[3] - ocr_box[1])

    # Calculate the area of the rectangular label box
    rect_area = (rect_label_box[2] - rect_label_box[0]) * (rect_label_box[3] - rect_label_box[1])

    # Calculate the percentage of intersection area with respect to OCR box area
    overlap_percentage = ((intersection_area * 100) / ocr_area) 
    
    # Check if the overlap percentage is greater than or equal to the threshold
    if overlap_percentage >= threshold:
        return True
    else:
        return False

def find_words_within_boxes_2(ocr_boxes, rect_label_boxes):
    result = {}

    for rect_label_box in rect_label_boxes:
        x1, y1, width, height, label = rect_label_box
        x2, y2 = x1 + width, y1 + height

        words_within_box = []

        for ocr_box in ocr_boxes:
            ocr_x, ocr_y, ocr_width, ocr_height, text = ocr_box

            if is_box_inside_2([x1, y1, x2, y2], [ocr_x, ocr_y, ocr_x + ocr_width, ocr_y + ocr_height], 70):
                words_within_box.append(text)

        result[tuple(rect_label_box)] = words_within_box

    return result

def find_words_within_boxes(ocr_boxes, rect_label_boxes):
    result = {}

    for rect_label_box in rect_label_boxes:
        x1, y1, width, height, label = rect_label_box
        x2, y2 = x1 + width, y1 + height

        words_within_box = []

        for ocr_box in ocr_boxes:
            ocr_x, ocr_y, ocr_width, ocr_height, text = ocr_box

            # Check if the OCR box is within the Rect Label Box
            if x1 <= ocr_x <= x2 and y1 <= ocr_y <= y2:
                words_within_box.append(text)

        result[tuple(rect_label_box)] = words_within_box

    return result

def get_bbox_words(json_op):
    result=[]
    for page in json_op['pages']:
        y,x=page['dimensions']
        for block in page['blocks']:
            for lines in block['lines']:
                for word in lines['words']:
                    ((xi,yi),(xj,yj))=word['geometry']
                    result.append([int(xi*image_width),int(yi*image_height),int((xj-xi)*image_width),int((yj-yi)*image_height),[word['value']]])
                    # print([xi*x,yi*y,xj*x-xi*x,yj*y-yi*y,[word['value']]])
                    # result['words'].append(word['value'])
    return result

# BASE FUNCTION TO PERFORM DOCTR PREDICTIONS
def predict(img_path):
    doc = DocumentFile.from_images(img_path)
    result = model(doc)
    output = get_bbox_words(result.export())
    return output


def subfinder(words_list, answer_list, max_mismatches=30):
    best_match_count = 0
    best_start_idx = 0
    best_end_idx = 0

    # Set the Max_Mismatches to be a percentage of the length of the words_list
    max_mismatches = int(len(answer_list) * max_mismatches / 100)

    for start_idx in range(len(words_list) - len(answer_list) + 1):
        match_count = 0
        for i in range(len(answer_list)):
            if words_list[start_idx + i] == answer_list[i]:
                match_count += 1

        mismatches = len(answer_list) - match_count
        if mismatches <= max_mismatches and match_count > best_match_count:
            best_match_count = match_count
            best_start_idx = start_idx
            best_end_idx = start_idx + len(answer_list) - 1

    if best_match_count > 0:
        return answer_list, best_start_idx, best_end_idx
    else:
        return None, 0, 0

def process_example(dataset, doc_id, qa_id, model_checkpoint="microsoft/layoutlmv2-large-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    question = dataset[doc_id]["q_and_a"][qa_id]["question"]
    answers = dataset[doc_id]["q_and_a"][qa_id]["answer"]
    words = dataset[doc_id]["words"]
    boxes = dataset[doc_id]["boxes"]

    answer_f = " ".join(answer[0] for answer in answers)
    answer_f1 = [answer_f]

    word_f = [word[0] for word in words]

    encoding = tokenizer(question, word_f, boxes=boxes, padding="max_length", max_length=512)
    sequence_ids = encoding.sequence_ids()

    # Sequence Id Length Check
    # print(len(sequence_ids))

    if len(encoding.input_ids) > 512:
        set_greater.add(dataset[doc_id]["file_name"])

    match, word_idx_start, word_idx_end = subfinder(word_f, answer_f.split())

    token_start_index = 0
    while sequence_ids[token_start_index] != 1:
        token_start_index += 1

    token_end_index = len(encoding.input_ids) - 1
    while sequence_ids[token_end_index] != 1:
        token_end_index -= 1

    word_ids = encoding.word_ids()[token_start_index:token_end_index+1]

    for id in word_ids:
        if id == word_idx_start:
            start_position = token_start_index
        else:
            token_start_index += 1

    for id in word_ids[::-1]:
        if id == word_idx_end:
            end_position = token_end_index
        else:
            token_end_index -= 1

    return start_position, end_position, encoding

input_files = os.listdir(input_files_folder)

for input_file in input_files:
    # Load the Original Dataset
    input_file = os.path.join(input_files_folder,input_file)
    print('Name of the Input File : ',input_file)
    original_dataset = json.load(open(input_file))

    print("Dataset Loaded!!!")

    # Load the labels dataset
    l_dataset = None

    # Drop everything other that id, annotations, data, and prediction
    original_dataset = [{k: entry[k] for k in ('id', 'annotations', 'data', 'predictions')} for entry in original_dataset]

    dataset_docvqa = []

    for i in original_dataset:
        dict_new = {}
        dataset_annotations = []

        for entry in i["annotations"]:
            count = 0
            for dict in entry["result"]:
                # Drop all entries with from_name as bbox
                if dict["from_name"] == "bbox":
                    count += 1
                else:
                    dataset_annotations.append(dict)
        # Update the dataset_docvqa with the file_name and the annotations
        dict_new["file_name"] = i["data"]['ocr'].split('/')[-1]
        dict_new["annotations"] = dataset_annotations

        dataset_docvqa.append(dict_new)

    l_dataset = dataset_docvqa

    # Dataset
    dataset = []

    # Enumerate
    for i, file_data in tqdm(enumerate(original_dataset)):
        # print(f"Processing {i}th file")

        try:
            entry_dict = {}
            file_path = file_data["data"]["ocr"].split("/")[-1]
            entry_dict["file_name"] = file_data["data"]["ocr"].split("/")[-1]
            # entry_dict["dimensions"] = [l_dataset[i]["annotations"][0]["original_width"], l_dataset[i]["annotations"][0]["original_height"]]
            entry_dict["dimensions"]=image_width,image_height

            # Get the OCR Boxes
            # ocr_boxes = extract_ocr_boxes(file_data, l_dataset[i]["annotations"][0]["original_width"], l_dataset[i]["annotations"][0]["original_height"])
            ocr_boxes=predict(os.path.join(img_dir,file_path))
            
            words = []
            boxes = []

            for box in ocr_boxes:
                x, y, width, height, text = box
                words.append(text)
                boxes.append([x, y, x + width, y + height])

            entry_dict["words"] = words
            entry_dict["boxes"] = boxes

            # Get the Rect Label Boxes
            rect_label_boxes = extract_rect_label_boxes(l_dataset[i], l_dataset[i]["annotations"][0]["original_width"], l_dataset[i]["annotations"][0]["original_height"])

            # Find the words within the boxes
            result = find_words_within_boxes_2(ocr_boxes, rect_label_boxes)

            q_and_a = []

            # Now, add all the questions and answers to the dictionary
            for rect_label_box, words_within_box in result.items():
                if rect_label_box[-1] not in label_to_question.keys():
                    continue
                else:
                    question = label_to_question[rect_label_box[-1]]
                q_and_a.append({
                    "question" : question,
                    "answer" : words_within_box
                })

            entry_dict["q_and_a"] = q_and_a

            dataset.append(entry_dict)
        except:
            continue

    # Save the dataset
    # with open("dataset_lp_v2-1.json", "w") as f:
        # json.dump(dataset, f, indent=4)



    set_greater = set()
    print('TOKENIZING')



    # Load the dataset
    # dataset = json.load(open("./dataset_lp_v2-1.json"))

    # Process the dataset
    dataset_v3 = dataset

    # Create a file to log errors
    error_log_file = open("error_log.txt", "w")

    for doc_id in tqdm(range(0, len(dataset))):
        for qa_id in range(0, len(dataset[doc_id]["q_and_a"])):
            try:
                start_idx, end_idx, encoding = process_example(dataset, doc_id, qa_id,model_checkpoint_v2)
                # Add the start and end indices to the dataset
                dataset[doc_id]["q_and_a"][qa_id]["start_idx"] = start_idx
                dataset[doc_id]["q_and_a"][qa_id]["end_idx"] = end_idx
                dataset[doc_id]["q_and_a"][qa_id]["input_ids"] = encoding.input_ids
            except Exception as e:
                # Log the error to the file
                error_log_file.write(f"Error processing Document {doc_id}, QA {qa_id}: {str(e)}\n")
            try:
                start_idx, end_idx, encoding = process_example(dataset_v3, doc_id, qa_id,model_checkpoint_v3)
                # Add the start and end indices to the dataset
                dataset_v3[doc_id]["q_and_a"][qa_id]["start_idx"] = start_idx
                dataset_v3[doc_id]["q_and_a"][qa_id]["end_idx"] = end_idx
                dataset_v3[doc_id]["q_and_a"][qa_id]["input_ids"] = encoding.input_ids
            except Exception as e:
                # Log the error to the file
                error_log_file.write(f"Error processing Document {doc_id}, QA {qa_id}: {str(e)}\n")

    # Close the error log file
    error_log_file.close()

    # Save the dataset
    with open(os.path.join(output_dir,os.path.basename(input_file).split('.')[0]+'_final_v2.json'), "w") as f:
        json.dump(dataset, f, indent=4)
    with open(os.path.join(output_dir,os.path.basename(input_file).split('.')[0]+'_final_v3.json'), "w") as f:
        json.dump(dataset_v3, f, indent=4)

    print('Data saved in :',output_dir)

print('ALL THE DATA FILES ARE PROCESSED.')
print('MERGING THE DATA FILES')

input_folder=output_dir
output_folder=output_dir

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

print('FINAL DATA FILES SAVED')