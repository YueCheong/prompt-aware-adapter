import os
import random
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )
        
        # zy add to construct instuction_input for cc_sbu
        self.instruction_pool = [
            'Briefly describe this image.',
            'Provide a concise depiction of this image.',
            'Present a short description of this image.',
            'Summarize this image in a few words.',
            'A short image caption:',
            'A short image description:',
            'A photo of ',
            'An image that shows ',
            'Write a short description for the image. ',
            'Write a description for the photo.',
            'Provide a description of what is presented in the photo.',
            'Briefly describe the content of the image.',
            'Can you briefly explain what you see in the image?',
            'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.',
            'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.',
        ]
        
    def to_dict(self, sample):
        # zy add to construct instuction_input for cc_sbu
        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)
        
        return {
            "image": sample[0],
            "answer": self.text_processor(sample[1]["caption"]),
            "instruction_input": instruction,
        }


class CCSBUAlignDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]
        # zy add to construct instuction_input for cc_sbu
        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)

        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instruction_input": instruction,
        }