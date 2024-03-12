import os
import torch

from src.helper import *

# Basic args for Tactful 
args = {
    "strategy":'fl2mi',      # strategy to be used for tactful
    "total_budget":1000,  # Total data points available
    "budget":20,  # Budget per iteration
    "lake_size":1278,  # Size of the lake dataset
    "train_size":10,  # Size of the training dataset
    "category":'Reference Block',   # Target Class     Note : use Stamps-Seals instead of Stamps/Seals due to path issues
    "device":0,   # GPU Device
    "proposal_budget":20,  # Budget for proposal generation
    "iterations":30,   # Total AL Rounds
    "batch_size":4    # Batch Size
}
args["output_path"] = args['strategy']

# Name of the model to be trained layoutlmv2/layoutlmv3
model_name = 'layoutlmv3'    # Name of the Model
learning_rate = 1e-5         # Learning Rate of the Model Training
epoch_t=1                    # The Epoch count after which the the model starts saving the sample question and answers for every 5 epochs
init_epochs=3                # The Epoch count of the Initial Training
al_epochs = 5                # The Epoch count of Each Active Learning round


# Requried data paths
train_path = '/data/circulars/DATA/layoutLM+Tactful/model_output7'    # path of the output dir
data_dir = '/data/circulars/DATA/layoutLM+Tactful/layoutlmv3_data' # path to the splitted data
query_path = '/data/circulars/DATA/layoutLM+Tactful/query_images'  # path to the query Images Folder
banned_txt_path = '/data/circulars/DATA/LayoutLM/docvqa_dataset/banned_txt_files.txt'  # path to the txt file consisting of the names of the banned files
init_model_path = 'microsoft/layoutlmv3-large'   # Initial model Path
full_data_annots = '/data/circulars/DATA/layoutLM+Tactful/full_data_annots.json'  # Path to the file consisting of the class wise annotations of the data

# Wandb Credentials
wandb_flag=1
wandb_project_desc=f'layoutlm_Tactful_FineTuning_{model_name}'
wandb_name=f'{os.path.basename(train_path)}_{args["strategy"]}_{learning_rate}_{args["batch_size"]}'
wandb_model_desc=f'{model_name} - Fine Tuned on DocVQA using layoutlm+tactful'
wandb_key='ead46cf543385050fcec224a0c2850faffcae584'

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
    # "Body Block" : "What is the body of the circular?",
    "Copy-Forwarded To Block" : "Who is the circular copied to?",
    "Reference Id" : "What is the reference id of the circular?",
    "Reference Block": "What is the reference mentioned in the circular?",
    "Circular ID": "What is the identification number of the circular?",
}

# Auto Initialization
MAPPING = {'0': 'Date Block', '1': 'Logos', '2': 'Subject Block', '3': 'Body Block', '4': 'Circular ID', '5': 'Table', '6': 'Stamps-Seals', '7': 'Handwritten Text', '8': 'Copy-Forwarded To Block', '9': 'Address of Issuing Authority', '10': 'Signature', '11': 'Reference Block', '12': 'Signature Block', '13': 'Header Block', '14': 'Addressed To Block'}

train_data_dirs = (os.path.join(data_dir,"train"),
                   os.path.join(data_dir,"docvqa_train.json"))
lake_data_dirs = (os.path.join(data_dir,"lake"),
                  os.path.join(data_dir,"docvqa_lake.json"))
val_data_dirs = (os.path.join(data_dir,"val"),
                 os.path.join(data_dir,"docvqa_val.json"))

query_path = os.path.join(query_path, args['category'])

training_name = args['output_path']
model_path = os.path.join(train_path, training_name)
if (not os.path.exists(model_path)):
    create_dir(model_path)
output_dir = os.path.join(model_path, "initial_training")

selection_arg = {"class":args['category'], 'eta':1, "model_path":model_path, 'smi_function':args['strategy']}

if torch.cuda.is_available():
    torch.cuda.set_device(args['device'])