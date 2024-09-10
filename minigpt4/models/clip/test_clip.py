import torch
# import clip
from model import CLIP
from clip import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load("/home/alisa/.cache/clip/ViT-B-32.pt", device=device)

prompts = ["A photo of a cat", "A photo of a dog", "A landscape photo"]  
text_inputs = tokenize(prompts).to(device)  


with torch.no_grad(): 
    text_features = model.encode_text(text_inputs)
    token_features = model.encode_token(text_inputs)

# text_features = text_features / text_features.norm(dim=-1, keepdim=True)
# text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

print("Encoded text features:", text_features.shape)
print(text_features)

print("Encoded text embeddings:", token_features.shape)
print(token_features)

