import os
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

# Basic args for Tactful 
args = {
    "strategy":'fl2mi',      # strategy to be used for tactful
    "total_budget":1000,  # Total data points available
    "budget":20,  # Budget per iteration
    "lake_size":1268,  # Size of the lake dataset
    "train_size":20,  # Size of the training dataset
    "category":'Reference Block',   # Target Class     Note : use Stamps-Seals instead of Stamps/Seals due to path issues
    "device":0,   # GPU Device
    "proposal_budget":10,  # Budget for proposal generation
    "iterations":100,   # Total AL Rounds
    "batch_size":4    # Batch Size
}


# Name of the model to be trained layoutlmv2/layoutlmv3
model_name = 'layoutlmv3'    # Name of the Model
learning_rate = 1e-5         # Learning Rate of the Model Training
epoch_t=1                    # The Epoch count after which the the model starts saving the sample question and answers for every 5 epochs
init_epochs = 30                # The Epoch count of the Initial Training
al_epochs = 100                # The Epoch count of Each Active Learning round
ROUGE_THRESH = 0.95          # Threshold for Training a Round BAsed on Rouge-L F1 Score
TRAIN_LOSS_THRESH = 0.4      # Threshold for Training a Round BAsed on Train Loss
EPOCH_THRESH = 1            # Threshold for Minimum Epoch in a Round
SCHEDULER_T_MAX = 30        # Steps for the Scheduler
SATURATION_THRESH = 5       # Minimum epochs for Saturated Train Loss


# Requried data paths
train_path = '/data/circulars/DATA/layoutLM+Tactful/model_output6_2'    # path of the output dir
data_dir = '/data/circulars/DATA/layoutLM+Tactful/layoutlmv3_data' # path to the splitted data
query_path = '/data/circulars/DATA/layoutLM+Tactful/query_images'  # path to the query Images Folder
banned_txt_path = '/data/circulars/DATA/LayoutLM/docvqa_dataset/banned_txt_files.txt'  # path to the txt file consisting of the names of the banned files
init_model_path = 'microsoft/layoutlmv3-large'   # Initial model Path
full_data_annots = '/data/circulars/DATA/layoutLM+Tactful/full_data_annots.json'  # Path to the file consisting of the class wise annotations of the data
images_dir = '/data/circulars/DATA/LayoutLM/docvqa_dataset/Images'   # Path to the Image Directory

# Wandb Credentials
wandb_flag=0
wandb_project_desc=f'layoutlm_Tactful_FineTuning_{model_name}'
wandb_name=f'{os.path.basename(train_path)}_{args["strategy"]}_{learning_rate}_{args["batch_size"]}_{ROUGE_THRESH}'
wandb_model_desc=f'{model_name} - Fine Tuned on DocVQA using layoutlm+tactful'
wandb_key='ead46cf543385050fcec224a0c2850faffcae584'

rcnn_model_path = '/data/circulars/DATA/TACTFUL/faster_rcnn_output/random/initial_training/model_final.pth'
config_path = '/data/circulars/DATA/TACTFUL/Data/faster_rcnn_pub_config.yml'

# Questions corresponding to different Classes
label_to_question = {
    "Date Block" : "What is the date when the circular was issued?",
    "Subject Block" : "What is the subject of the circular?",
    "Adressed To" : "To whom is the circular addressed to?",
    "Addressed To Block": "What is the complete address of the recipient?",
    "Address of Issuing Authority": "What is the address of the issuing authority?",
    "Address Block" : "What is the complete address of the circular?",
    "Header Block" : "What is the header of the circular?",
    "Signature Block" : "Who is the signatory of the circular?",
    "Copy-Forwarded To Block" : "Who is the circular copied to?",
    "Reference Id" : "What is the reference id of the circular?",
    "Reference Block": "What is the reference mentioned in the circular?",
    "Circular ID": "What is the identification number of the circular?",
}

MAPPING = {'Address of Issuing Authority': 0, 'Date Block': 1, 'Header Block': 2, 'Table': 3, 'Circular ID': 4, 'Body Block': 5, 'Signature': 6, 'Signature Block': 7, 'Stamps-Seals': 8, 'Handwritten Text': 9, 'Copy-Forwarded To Block': 10, 'Addressed To Block': 11, 'Subject Block': 12, 'Logos': 13, 'Reference Block': 14, 'Adressed To': 15, 'Circular Reference': 16, 'Name of the signatory': 17, 'Signatory-Designation': 18, 'Reference Id': 19, 'Forwarder': 20, 'Forwarder-Designation': 21, 'Issuing Authority': 22}
R_MAPPING = {str(value):key for key,value in MAPPING.items()}

train_data_dirs = (os.path.join(data_dir,"train"),
                   os.path.join(data_dir,"docvqa_train.json"))
lake_data_dirs = (os.path.join(data_dir,"lake"),
                  os.path.join(data_dir,"docvqa_lake.json"))
val_data_dirs = (os.path.join(data_dir,"val"),
                 os.path.join(data_dir,"docvqa_val.json"))

query_path = os.path.join(query_path, args['category'])

args["output_path"] = args['strategy']
training_name = args['output_path']
model_path = os.path.join(train_path, training_name)
def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass
if (not os.path.exists(model_path)):
    create_dir(model_path)
output_dir = os.path.join(model_path, "initial_training")

selection_arg = {"class":args['category'], 'eta':1, "model_path":model_path, 'smi_function':args['strategy']}

if torch.cuda.is_available():
    torch.cuda.set_device(args['device'])

cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MAPPING)
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.IMS_PER_BATCH = 10
cfg.MODEL.RPN.NMS_THRESH = 0.8
cfg.MODEL.RPN.POST_NMS_TOPK_TEST= 2000
cfg.OUTPUT_DIR = output_dir
cfg.TRAINING_NAME = training_name
cfg.MODEL.WEIGHTS = rcnn_model_path