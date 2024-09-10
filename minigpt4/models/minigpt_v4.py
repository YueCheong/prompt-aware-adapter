"""
minigpt_v4
1. no glocal-attn 
2. local-attn use (n) token_features from LLM 
"""

import re

import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
from minigpt4.models.local_attn import LocalAttention, MultiHeadLocalAttention
# from minigpt4.visualization.visualization import visualize_local_attn


@registry.register_model("minigpt_v4")
class MiniGPTv4(MiniGPTBase):
    """
    MiniGPT-v4 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigptv4_llama3.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            llama_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        self.local_attn = LocalAttention(
            self.visual_encoder.num_features, self.llama_model.config.hidden_size, self.visual_encoder.num_features
        )
        
        img_f_dim = self.visual_encoder.num_features * 4
        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )
        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()
        
        # zy add for visualization 
        self.norm_dots = None
        self.local_attns = None   

    def encode_img(self, image):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

            inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def encode_img_txt(self, image, prompt):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])
            
        with self.maybe_autocast():     
            # get image embedding
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_embeds = image_embeds[:, 1:, :]
            
            prompt_aware_image_embeds = None 

            # get prompt embedding
            if prompt:
                task_flag = re.compile(r'\[.*\]')
                prompt_list = [task_flag.split(p)[-1].strip() if re.search(task_flag, p) else p for p in prompt]
                
                self.llama_tokenizer.padding_side = "right"
                prompt_tokens = self.llama_tokenizer(
                    prompt_list,
                    return_tensors="pt",
                    padding="longest",
                    add_special_tokens=False
                ).to(device)
                
                prompt_embeds = self.embed_tokens(prompt_tokens.input_ids.long()).to(device) # zy modify to avoid Float error # [b,l,d] [6,48,4096]   
                
                prompt_aware_image_embeds, norm_dots, local_attns = self.local_attn(images=image_embeds, texts=prompt_embeds, mask=None) # [6, 1024, 1408]
                self.norm_dots = norm_dots
                self.local_attns = local_attns
   
            # control_image_embeds = self.res_ratio * prompt_aware_image_embeds + (1 - self.res_ratio) * image_embeds # [6, 1025, 1408]
            control_image_embeds = prompt_aware_image_embeds if prompt_aware_image_embeds is not None else image_embeds              
            bs, pn, hs = control_image_embeds.shape
            control_image_embeds = control_image_embeds.view(bs, int(pn / 4), int(hs * 4))  # -> [6, 256, 5632]   

            inputs_llama = self.llama_proj(control_image_embeds) # [6, 256, 4096]  
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device) # [6, 256]
        return inputs_llama, atts_llama

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Minigpt-4-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        print(f'model arch:\n {model}')
        return model
