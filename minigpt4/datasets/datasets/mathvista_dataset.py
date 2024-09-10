from datasets import load_dataset
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class MathVistaDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path=None):
        """
        Initialize the MathVista dataset loader.
        
        Args:           
            vis_root (string): Root directory of images (e.g. coco/images/)
            ann_root (string): directory to store the annotation file
        """
        self.dataset = load_dataset("AI4Math/MathVista")['testmini'] # testmini, test
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vis_root = vis_root
        self.ann_path = ann_path
        

    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Retrieve the dataset item at the specified index."""
        item = self.dataset[index]
        
        # Process image
        image_path = os.path.join(self.vis_root, item['image'])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(item['question'])
        answer = self.text_processor(item['answer'])
                               
        question = "<Img><ImageHere></Img> [vqa] {} ".format(question)
   
        return {
            "pid": item['pid'],
            "image": image,
            "query": item['query'],
            "instruction_input": question,
            "answer": answer
        }


if __name__ == "__main__":
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset =  MathVistaDataset(split='testmini', vis_root='/home/users/nus/idmwyk/scratch/temp/dataset/MathVista', transform=transform)
    mathvista = DataLoader(dataset, batch_size=2, shuffle=False) 
    
    for dict in mathvista:
        pid = dict['pid']
        image = dict['image']
        question = dict['question'] 
        answer = dict['answer']
        query = dict['query']
        print(f'image_id:{pid}\nimage:{image}\nquestion:{question}\nanswer:{answer}\nquery:{query}')
        break   


# from datasets import load_dataset

# dataset = load_dataset("AI4Math/MathVista")

# # print the first example on the testmini set
# print(dataset["testmini"][0])
# print(dataset["testmini"][0]['pid']) # print the problem id 
# print(dataset["testmini"][0]['question']) # print the question text 
# print(dataset["testmini"][0]['query']) # print the query text
# print(dataset["testmini"][0]['image']) # print the image path
# print(dataset["testmini"][0]['answer']) # print the answer
# dataset["testmini"][0]['decoded_image'] # display the image

# # print the first example on the test set
# print(dataset["test"][0])


