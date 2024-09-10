"""
ZY add COCOVRDataset
Visual Reasoning, include 4 types of quesitons: object, number, color, location
"""

import os
import json
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class COCOVRDataset(Dataset): # ZY
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
                      
        self.vis_processor = vis_processor    
        self.text_processor = text_processor
        self.vis_root = vis_root            
        self.img_id_file = os.path.join(ann_path, "img_ids.txt")
        self.question_file = os.path.join(ann_path, "questions.txt")
        self.answer_file = os.path.join(ann_path, "answers.txt")
        self.type_file = os.path.join(ann_path, "types.txt")

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]
      
        with open(self.img_id_file, 'r') as f:
            self.img_ids = f.read().splitlines()
        with open(self.question_file, 'r') as f:
            self.questions = f.read().splitlines()
        with open(self.answer_file, 'r') as f:
            self.answers = f.read().splitlines()
        with open(self.type_file, 'r') as f:
            self.types = f.read().splitlines()      
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
               
        image_file = 'COCO_train2014_{:0>12}.jpg'.format(img_id) 
        image_path = os.path.join(os.path.join(self.vis_root, 'train2014'), image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)           # [C, H, W] = [3, 224, 224]

        question = self.text_processor(self.questions[index])
        answer = self.text_processor(self.answers[index])
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        
        type = self.types[index]
        
        return {
            "image_id": img_id,                     
            "image": image,
            "instruction_input": instruction, 
            "answer": answer,
            "type": type
        }
        

class COCOVREvalDataset(Dataset): # ZY
    def __init__(self, vis_processor, root_path):
        self.vis_processor = vis_processor    
        self.vis_root = os.path.join(root_path, 'images' )          
        self.img_id_file = os.path.join(root_path, "annotations/test/img_ids.txt")
        self.question_file = os.path.join(root_path, "annotations/test/questions.txt")
        self.answer_file = os.path.join(root_path, "annotations/test/answers.txt")
        self.type_file = os.path.join(root_path, "annotations/test/types.txt")

        # self.instruction_pool =[
        #     "[vqa] {}",
        #     "[vqa] Based on the image, respond to this question with a short answer: {}"
        # ]
      
        with open(self.img_id_file, 'r') as f:
            self.img_ids = f.read().splitlines()
        with open(self.question_file, 'r') as f:
            self.questions = f.read().splitlines()
        with open(self.answer_file, 'r') as f:
            self.answers = f.read().splitlines()
        with open(self.type_file, 'r') as f:
            self.types = f.read().splitlines()      
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
               
        image_file = 'COCO_val2014_{:0>12}.jpg'.format(img_id) 
        image_path = os.path.join(os.path.join(self.vis_root, 'val2014'), image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)           # [C, H, W] = [3, 224, 224]
        
        question = '[vqa]' + self.questions[index]
        answer = self.answers[index]
        # instruction = random.choice(self.instruction_pool).format(question)
        # instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        type = self.types[index]
        
        return img_id, image, question, answer, type

     
     
if __name__ == "__main__":
    # If u want to test the COCOVRDataset, 
    # comment out the "self.vis_processor" and "self.text_procesor" part first
    
    dataset = COCOVRDataset(vis_processor=0, text_processor=0, vis_root='/ssd/data0/alisa/dataset/coco_vr/images',
                            ann_path = '/ssd/data0/alisa/dataset/coco_vr/annotations/train') 
    coco_vr = DataLoader(dataset, batch_size=2, shuffle=False) 
        
    for dict in coco_vr:
        image_id = dict['image_id']
        image = dict['image']
        instruction_input = dict['instruction_input'] 
        answer = dict['answer']
        type = dict['type']
        print(f'image_id:{image_id}\ninstruction_input:{instruction_input}\nanswer:{answer}\ntype{type}')
        break           
        