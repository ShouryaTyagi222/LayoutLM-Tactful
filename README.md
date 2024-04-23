# LayoutLM+Tactful
## Required Data 
- Query Images - Folders of Images divided based on Classes.
- Splitted Data  - Splitted Train, lake and Val Dataset.(Images and Required input to the layoutlm model)
- Class Wise Annotations - Class Annotations Details of Every Image (Classes along with their bounding box information for every image).
- Banned_txt_file - a txt file consisting of the names of the files which are to be avoided for the training.

## Data Preparation
To Prepare Query Image , run query_gen.py

```python query_gen.py -i FULL_IMAGE_FOLDER -c FOLDER_CONSISTING_OF_RAW_ANNOTATED_FILES  -q OUTPUT_FOLDER_PATH```

The Following will generate a folder of the folders of query Images on the Basis of classes.

To Prepare Data - Preprocess the RAW_DATA_FILES by running data/prepare_data.py
```Adjust the tokenizer_checkpoints
Input_file_folder : path to the folder of raw Data files.
Img_dir : Path to the Full Images Folder
Output_dir : Path to the output Images folder.
Image_width & Image_height
```
The following will generate a separate processed file for each raw data file for both layoutlmV2 and layoutlmv3.
If the data files are required to be merged manually use data/merge_json_data.py

Then to split the data into train, val and lake run split_data.py
```
python split_data.py -i  PREPARED_JSON_FILE -d FULL_IMAGES_FOLDER -o OUTPUT_FOLDER
```
The Following will divide the Data into Train, Val and lake set in the given output folder.
To Prepare the class Wise Annotations File , run prepare_annot.py
```
python prepare_annots.py -i FULL_IMAGE_FOLDER -c FOLDER_CONSISTING_OF_RAW_ANNOTATED_FILES -q OUTPUT_FOLDER_PATH
```
The Following will generate data file for the class wise annotations.



## Training 
- Model Configurations : 
1. Basic Configuration
2. Strategy - The strategy used for tactful active learning.
3. Total Budget - Total data points available for training.
4. Budget - Budget per iteration for selecting data points.
5. Lake Size - Size of the lake dataset (unlabeled data).
6. Train Size - Size of the initial training dataset.
7. Category - Target class for training.
8. Device - GPU Device to be used.
9. Proposal Budget - Budget for proposal generation in each iteration.Iterations: 30 - Total active learning rounds.
10. Batch Size - Batch size used during training.
- Model Configuration
1. Model Name - Name of the model to be trained (layoutlmv2 or layoutlmv3).
2. Learning Rate - Learning rate used for model training.
3. ROUGE_THRESH - Threshold for Training a Round BAsed on Rouge-L F1 Score
4. TRAIN_LOSS_THRESH - Threshold for Training a Round BAsed on Train Loss
5. EPOCH_THRESH - Threshold for Minimum Epoch in a Round
6. SCHEDULER_T_MAX - Steps for the Scheduler
- Data Paths
1. Train Path - Path of the output directory.
2. Data Directory - Path to the splitted data directory.
3. Query Paths - Path to the query images folder.
4. Banned Text Path - Path to the text file consisting of banned - files names.
1. Initial Model Path - Path to the initial pre-trained model.
2. Full Data Annotations - Path to the file containing class-wise annotations of the data.
- Wandb (Weights & Biases) 
1. ConfigurationWandb Flag - Flag indicating whether to use Wandb for logging.
2. Wandb Project Description - Description of the project in Wandb.
3. Wandb Name - Name for Wandb run.
4. Wandb Model Description - Description of the model for Wandb.
5. Wandb Key: API key for Wandb.Questions for Different Classes
- Label to Question: Mapping between class labels and corresponding questions for model evaluation.

	After Adjusting the configurations you can start the Training,	
		```
		python train.py
		```


## Author
1. Shourya Tyagi (Intern)