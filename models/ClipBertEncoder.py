import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel

class ClipBertEncoder(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()

        self.img_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # multimodal but we only use the image part
        self.img_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1') # for the title
        
        
        self.descmodel = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        
        # Get dimensions
        self.img_dim = 512
        self.title_dim = 512 
        self.desc_dim = 768
        
        # Freeze backbone models if specified
        if frozen:
            for param in self.mclip_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Project description embeddings to match CLIP dimensions
        self.desc_projector = nn.Linear(self.desc_dim, self.img_dim)
        
        # Fusion layer to combine all three embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.img_dim * 3, self.img_dim),
            nn.ReLU()
        )
        
    def encode_image(self, images):
        inputs = self.img_processor(images=images, return_tensors="pt")
        img_feat = self.img_model.get_image_features(**inputs)
        return img_feat
    
    def encode_title(self, titles):
        encoded_titles = self.text_model.encode(titles)
        return encoded_titles
    
    def encode_description(self, descriptions):

        encoded_input = self.tokenizer(descriptions, return_tensors='pt')
        output = self.descmodel(**encoded_input)

        desc_features = output.last_hidden_state.mean(dim=1)
        desc_features = self.desc_projector(desc_features)
        return desc_features
    
    def forward(self, x):
        
        img_features = self.encode_image(x["image"])
        title_features = self.encode_title(x["title"])
        desc_features = self.encode_description(x["description"])

        # Normalize features
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        title_features = title_features / title_features.norm(dim=1, keepdim=True)
        desc_features = desc_features / desc_features.norm(dim=1, keepdim=True)

        # Concatenate all features
        combined_features = torch.cat([img_features, title_features, desc_features], dim=1)
        
        # Fuse the features
        fused = self.fusion_layer(combined_features)

        return fused