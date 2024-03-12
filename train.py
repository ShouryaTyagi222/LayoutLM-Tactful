# python train.py -i
import os
import torch
import argparse

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import AdamW

import numpy as np
import logging
import wandb

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import CosineAnnealingLR
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from src.tactful_smi import TACTFUL_SMI
from src.helper import *
from configs import *

print('RUNNING')


def main():
    iteration = args['iterations']
    selection_strag = args['strategy']
    selection_budget = args['budget']
    budget = args['total_budget']
    proposal_budget = args['proposal_budget']
    target_question = label_to_question[args['category']]
    print('target question :',target_question)

    if wandb_flag:
        wandb.init(project=wandb_project_desc, name=wandb_name)
        wandb.login(key=wandb_key)
        wandb.config.update({"learning_rate": learning_rate, "batch_size": args['batch_size'], "num_epochs": iteration, "model": wandb_model_desc})
        # Log in to WandB
        wandb.login()

    print('STRATEGY :', selection_strag)

    tokenizer = AutoTokenizer.from_pretrained(init_model_path)
            
    i = 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{args['device']}")
    smoother = SmoothingFunction()
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    # model.to(device)
    # scheduler = CosineAnnealingLR(optimizer, T_max=iteration)

    with open(os.path.join(model_path,f"{model_name}_logs.txt"), "w") as f:
        f.write("")

    # Initial Training
    if not os.path.exists(os.path.join(output_dir,'model')):
        print('Loading the Initial Model')
        model = AutoModelForQuestionAnswering.from_pretrained(init_model_path)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        model.to(device)

        train_dataloader=load_data(train_data_dirs[1],train_data_dirs[0], args['batch_size'], banned_txt_path)
        test_dataloader=load_data(val_data_dirs[1],val_data_dirs[0], args['batch_size'], banned_txt_path)

        # del l_model
        torch.cuda.empty_cache()

        model.train()
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write("\nStarting the Initial Training\n\n")

        print('Training the Initial Model')
        epoch=0
        rougel_f_tr=0
        prev_test_loss = 0
        curr_test_loss = 1
        scheduler = CosineAnnealingLR(optimizer, T_max=iteration)
        # for epoch in range(init_epochs):  # loop over the dataset multiple times

        while rougel_f_tr<=0.75 and prev_test_loss!=curr_test_loss:
            model.train()
            progbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}, Train Loss = 0, current loss = 0 , Bleu Score = 0', unit='batch')
            Loss=[]
            bleu_scores=[0]
            o_answers=[]
            p_answers=[]
            questions=[]
            print(f'Epoch : {epoch+1}, learning rate : {optimizer.param_groups[0]["lr"]}')
            for batch in progbar:
                # get the inputs;
                input_ids = batch["input_ids"].to(device)
                question=batch['question'][0]
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                outputs = get_output(batch,model,model_name,device)

                # zero the parameter gradients
                optimizer.zero_grad()

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                start_pos = start_positions[0].item()
                end_pos = end_positions[0].item()
                ori_answer = input_ids[0][start_pos : end_pos + 1]
                ori_answer = tokenizer.decode(ori_answer)
                
                # Predicted Answer
                start_index = torch.argmax(outputs.start_logits[0], dim=-1).item()
                end_index = torch.argmax(outputs.end_logits[0], dim=-1).item()

                # Slice the input_ids tensor using scalar indices
                if start_index < end_index:
                    p_answer = input_ids[0][start_index:end_index + 1]
                else:
                    p_answer = input_ids[0][end_index:start_index + 1]

                # print('original answer :',ori_answer)

                # Decode the predicted answer
                p_answer = tokenizer.decode(p_answer)
                # print('predicted answer :',p_answer)

                bleu_score = corpus_bleu([ori_answer.split()], [p_answer.split()], smoothing_function=smoother.method1)
                bleu_scores.append(bleu_score)

                o_answers.append(ori_answer)
                p_answers.append(p_answer)
                questions.append(question)

                Loss.append(loss.item())
                progbar.set_description("Epoch : %s, Train Loss : %0.3f, current Loss : %0.3f, BLEU Score : %0.3f," % (epoch+1, np.mean(Loss), loss.item(), np.mean(bleu_scores)))

            rouge1_p, rouge1_r, rouge1_f, rouge2_p, rouge2_r, rouge2_f, rougel_p, rougel_r, rougel_f, target_rougel_f = cal_rouge_scores(target_question, questions, o_answers, p_answers)
            rougel_f_tr = rougel_f

            print(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f}, Target Class Rouge-l F1 Score: {target_rougel_f},\n')

            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write("Epoch = %s Train Loss = %0.3f, BLEU Score = %0.3f, Learning Rate = %s \n" % (epoch+1, np.mean(Loss), np.mean(bleu_scores), optimizer.param_groups[0]["lr"]))
            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f}, Target Class Rouge-l F1 Score: {target_rougel_f},\n')

            print(f'Epoch : {epoch+1}, learning rate : {optimizer.param_groups[0]["lr"]}')

            train_loss=np.mean(Loss)
            scheduler.step()

            model.eval()
            Loss = []
            bleu_scores=[]
            o_answers=[]
            p_answers=[]
            questions=[]
            progbar = tqdm(test_dataloader, desc=f'Epoch {epoch}', unit='batch')
            for batch in progbar:
                # get the inputs;
                input_ids = batch["input_ids"].to(device)
                question=batch['question'][0]
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                outputs = get_output(batch,model,model_name,device)

                loss = outputs.loss

                start_pos = start_positions[0].item()
                end_pos = end_positions[0].item()
                ori_answer = input_ids[0][start_pos : end_pos + 1]
                ori_answer = tokenizer.decode(ori_answer)
                # Predicted Answer
                start_index = torch.argmax(outputs.start_logits[0], dim=-1).item()
                end_index = torch.argmax(outputs.end_logits[0], dim=-1).item()

                # Slice the input_ids tensor using scalar indices
                if start_index < end_index:
                    p_answer = input_ids[0][start_index:end_index + 1]
                else:
                    p_answer = input_ids[0][end_index:start_index + 1]

                # Decode the predicted answer
                p_answer = tokenizer.decode(p_answer)

                bleu_score = corpus_bleu([ori_answer.split()], [p_answer.split()], smoothing_function=smoother.method1)
                o_answers.append(ori_answer)
                p_answers.append(p_answer)
                questions.append(question)
                Loss.append(loss.item())
                bleu_scores.append(bleu_score)
                progbar.set_description("Epoch : %s Test Loss : %0.3f, BLEU Score : %0.3f," % (epoch+1, np.mean(Loss), np.mean(bleu_scores)))

            rouge1_p, rouge1_r, rouge1_f, rouge2_p, rouge2_r, rouge2_f, rougel_p, rougel_r, rougel_f, target_rougel_f = cal_rouge_scores(target_question, questions, o_answers, p_answers)

            print(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f}, Target Class Rouge-l F1 Score: {target_rougel_f}, ')

            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write("Epoch = %s Test Loss = %0.3f, BLEU Score = %0.3f, Learning Rate = %s \n" % (epoch+1, np.mean(Loss), np.mean(bleu_scores), optimizer.param_groups[0]["lr"]))
            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f}, Target Class Rouge-l F1 Score: {target_rougel_f},\n')

            if wandb_flag:
                wandb.log({
                    "Training Loss": train_loss,
                    "Testing Loss": np.mean(Loss),
                    "Bleu Score":  np.mean(bleu_scores),
                    "Rouge-1 Recall": rouge1_p,
                    "Rouge-1 Precision": rouge1_r,
                    "Rouge-1 F1 Score": rouge1_f,
                    "Rouge-2 Recall": rouge2_p,
                    "Rouge-2 Precision": rouge2_r,
                    "Rouge-2 F1 Score": rouge2_f,
                    "Rouge-l Recall":rougel_p,
                    "Rouge-l Precision": rougel_r,
                    "Rouge-l F1 Score": rougel_f,
                    "Target Class Rouge-l F1 Score": target_rougel_f,
                })
            epoch+=1
            prev_test_loss = curr_test_loss
            curr_test_loss = np.mean(Loss)

            if epoch+1>=epoch_t and epoch%5==0:
                write_predictions_to_txt(questions, o_answers, p_answers, os.path.join(model_path,f"{model_name}_sample_q_and_a.txt"))

        model.save_pretrained(os.path.join(output_dir,'model'))
    else:
        print('Loading the Saved Model')
        try:
            model = AutoModelForQuestionAnswering.from_pretrained(os.path.join(output_dir,'model'))
            model.to(device)
        except:
            print('The Saved model is not found.')
            return 



    # Starting the AL Round

    i=0

    while (i < iteration and budget > 0):
        print(f'AL Round : {i}/{iteration}')
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
            f.write(f'\nAL Round : {i}/{iteration}\n')
        if (selection_strag != "random"):
            # creating new query set for under performing class for each iteration
            remove_dir(os.path.join(model_path, "query_images"))
            try:
                os.remove(os.path.join(model_path, "data_query.csv"))
            except:
                pass

            # Cropping object based on ground truth for the query set.
            # The set is part of train set, so no need of using object detection model to find the bounding box.
            print('>>>',query_path)
            print('>>>',os.path.join(model_path,'query_images'))
            crop_images_classwise_ground_truth(full_data_annots, query_path, os.path.join(
                model_path, "query_images"), args['category'])
            
            print("----------Crop Images Classwise ground truth DONE-------")

            remove_dir(os.path.join(model_path, "lake_images"))
            try:
                os.remove(os.path.join(model_path, "data.csv"))
            except:
                pass
            #l_model = create_model(cfg,'test')
            crop_images_classwise(
                full_data_annots, lake_data_dirs[0], os.path.join(model_path, "lake_images"), proposal_budget=proposal_budget)
            
            print("----------Crop Images Classwise DONE-------")

            selection_arg['iteration'] = i
            strategy_sel = TACTFUL_SMI(args = selection_arg)
            lake_image_list, subset_result = strategy_sel.select(proposal_budget)
            print('LENGTH OF SUBSET RESULT :',len(subset_result))
            subset_result = [lake_image_list[i] for i in subset_result]
            subset_result = list(
                set(get_original_images_path(subset_result,lake_data_dirs[0])))

        else:
            lake_image_list = os.listdir(lake_data_dirs[0])
            subset_result = Random_wrapper(
                lake_image_list, selection_budget)

        # reducing the selection budget
        budget -= len(subset_result)
        if (budget > 0):

            # transferring images from lake set to train set
            aug_train_subset(subset_result, train_data_dirs[1], lake_data_dirs[1], budget, lake_data_dirs, train_data_dirs)
            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write(f'LENGTH OF THE TRAIN DATA : {len(os.listdir(train_data_dirs[0]))}\n')
            print('LENGTH OF THE TRAIN DATA : ',len(os.listdir(train_data_dirs[0])))
           
        # removing the old training images from the detectron configuration and adding new one
        train_dataloader=load_data(train_data_dirs[1],train_data_dirs[0], args['batch_size'], banned_txt_path)
        test_dataloader=load_data(val_data_dirs[1],val_data_dirs[0], args['batch_size'], banned_txt_path)

        # del l_model
        torch.cuda.empty_cache()

        model.train()
        e=0
        rougel_f_tr=0
        prev_test_loss = 0
        curr_test_loss = 1
        # for epoch in range(init_epochs):  # loop over the dataset multiple times

        while rougel_f_tr<=0.8 and prev_test_loss!=curr_test_loss:
            progbar = tqdm(train_dataloader, desc=f'Epoch {e+1}, Train Loss = 0, current loss = 0 , Bleu Score = 0', unit='batch')
            Loss=[]
            bleu_scores=[0]
            o_answers=[]
            p_answers=[]
            questions=[]
            print(f'Epoch : {e}, learning rate : {optimizer.param_groups[0]["lr"]}')
            for batch in progbar:
                # get the inputs;
                input_ids = batch["input_ids"].to(device)
                question=batch['question'][0]
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                outputs = get_output(batch,model,model_name,device)

                # zero the parameter gradients
                optimizer.zero_grad()

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                start_pos = start_positions[0].item()
                end_pos = end_positions[0].item()
                ori_answer = input_ids[0][start_pos : end_pos + 1]
                ori_answer = tokenizer.decode(ori_answer)
                
                # Predicted Answer
                start_index = torch.argmax(outputs.start_logits[0], dim=-1).item()
                end_index = torch.argmax(outputs.end_logits[0], dim=-1).item()

                # Slice the input_ids tensor using scalar indices
                if start_index < end_index:
                    p_answer = input_ids[0][start_index:end_index + 1]
                else:
                    p_answer = input_ids[0][end_index:start_index + 1]


                # Decode the predicted answer
                p_answer = tokenizer.decode(p_answer)

                bleu_score = corpus_bleu([ori_answer.split()], [p_answer.split()], smoothing_function=smoother.method1)
                bleu_scores.append(bleu_score)

                o_answers.append(ori_answer)
                p_answers.append(p_answer)
                questions.append(question)

                Loss.append(loss.item())
                progbar.set_description("Epoch : %s, Train Loss : %0.3f, current Loss : %0.3f, BLEU Score : %0.3f," % (e+1, np.mean(Loss), loss.item(), np.mean(bleu_scores)))

            rouge1_p, rouge1_r, rouge1_f, rouge2_p, rouge2_r, rouge2_f, rougel_p, rougel_r, rougel_f, target_rougel_f = cal_rouge_scores(target_question, questions, o_answers, p_answers)
            rougel_f_tr = rougel_f

            print(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f}, Target Class Rouge-l F1 Score: {target_rougel_f},')

            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write("Epoch = %s Train Loss = %0.3f, BLEU Score = %0.3f, Learning Rate = %s \n" % (e+1, np.mean(Loss), np.mean(bleu_scores), optimizer.param_groups[0]["lr"]))
            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write(f'Train : rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f}, Target Class Rouge-l F1 Score: {target_rougel_f},\n')

            print(f'Epoch : {e+1}, learning rate : {optimizer.param_groups[0]["lr"]}')

            scheduler.step()
            train_loss=np.mean(Loss)

            model.eval()
            Loss = []
            bleu_scores=[]
            o_answers=[]
            p_answers=[]
            questions=[]
            progbar = tqdm(test_dataloader, desc=f'Epoch {e+1}', unit='batch')
            for batch in progbar:
                # get the inputs;
                input_ids = batch["input_ids"].to(device)
                question=batch['question'][0]
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                # # forward + backward + optimize
                outputs = get_output(batch,model,model_name,device)

                loss = outputs.loss

                start_pos = start_positions[0].item()
                end_pos = end_positions[0].item()
                ori_answer = input_ids[0][start_pos : end_pos + 1]
                ori_answer = tokenizer.decode(ori_answer)
                # Predicted Answer
                start_index = torch.argmax(outputs.start_logits[0], dim=-1).item()
                end_index = torch.argmax(outputs.end_logits[0], dim=-1).item()

                # Slice the input_ids tensor using scalar indices
                if start_index < end_index:
                    p_answer = input_ids[0][start_index:end_index + 1]
                else:
                    p_answer = input_ids[0][end_index:start_index + 1]

                # Decode the predicted answer
                p_answer = tokenizer.decode(p_answer)

                bleu_score = corpus_bleu([ori_answer.split()], [p_answer.split()], smoothing_function=smoother.method1)
                o_answers.append(ori_answer)
                p_answers.append(p_answer)
                questions.append(question)
                Loss.append(loss.item())
                bleu_scores.append(bleu_score)
                progbar.set_description("Epoch : %s Test Loss : %0.3f, BLEU Score : %0.3f," % (e+1, np.mean(Loss), np.mean(bleu_scores)))

            rouge1_p, rouge1_r, rouge1_f, rouge2_p, rouge2_r, rouge2_f, rougel_p, rougel_r, rougel_f, target_rougel_f = cal_rouge_scores(target_question, questions, o_answers, p_answers)

            print(f'rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f}, Target Class Rouge-l F1 Score: {target_rougel_f},')

            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write("Epoch = %s Test Loss = %0.3f, BLEU Score = %0.3f, Learning Rate = %s \n" % (e+1, np.mean(Loss), np.mean(bleu_scores), optimizer.param_groups[0]["lr"]))
            with open(os.path.join(model_path,f"{model_name}_logs.txt"), "a") as f:
                f.write(f'Test : rouge1_precision : {rouge1_p}, rouge1_recall : {rouge1_r}, rouge2_f1 : {rouge2_f}, rouge2_precision : {rouge2_p}, rouge2_recall : {rouge2_r}, rouge2_f1 : {rougel_f}, rougel_precision : {rougel_p}, rougel_recall : {rougel_r}, rougel_f1 : {rougel_f}, Target Class Rouge-l F1 Score: {target_rougel_f},\n')

            if wandb_flag:
                wandb.log({
                    "Training Loss": train_loss,
                    "Testing Loss": np.mean(Loss),
                    "Bleu Score":  np.mean(bleu_scores),
                    "Rouge-1 Recall": rouge1_p,
                    "Rouge-1 Precision": rouge1_r,
                    "Rouge-1 F1 Score": rouge1_f,
                    "Rouge-2 Recall": rouge2_p,
                    "Rouge-2 Precision": rouge2_r,
                    "Rouge-2 F1 Score": rouge2_f,
                    "Rouge-l Recall":rougel_p,
                    "Rouge-l Precision": rougel_r,
                    "Rouge-l F1 Score": rougel_f,
                    "Target Class Rouge-l F1 Score": target_rougel_f,
                    
                })

            e+=1
            prev_test_loss = curr_test_loss
            curr_test_loss = np.mean(Loss)



        if i+1>=epoch_t and i%5==0:
            write_predictions_to_txt(questions, o_answers, p_answers, os.path.join(model_path,f"{model_name}_sample_q_and_a.txt"))

        torch.cuda.empty_cache()

        i += 1
        print("remaining_budget", budget)
    model.save_pretrained(os.path.join(output_dir,'model'))


if __name__ == "__main__":
    main()