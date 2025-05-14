import torch
import torch.nn as nn
import clip
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

class ClipBertEncoder(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()

        self.mclip_model = CLIPModel.from_pretrained("M-CLIP/M-BERT-Base-ViT-B") 
        self.mclip_processor = CLIPProcessor.from_pretrained("M-CLIP/M-BERT-Base-ViT-B")
        
        
        #self.clip_model, self.clip_preprocess = clip.load("ViT-B/32") # for image and title processing
        
        
        self.text_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        # self.text_model = AutoModel.from_pretrained("distilbert-base-uncased") # for longer descriptions
        # self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Get dimensions
        self.img_dim = self.mclip_model.vision_model.config.hidden_size  # 768 for ViT-B
        self.title_dim = self.mclip_model.text_model.config.hidden_size  # 768 for M-BERT
        self.desc_dim = self.text_model.config.hidden_size  # 768 for BERT
        
        # Freeze backbone models if specified
        if frozen:
            for param in self.clip_model.parameters():
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
        
    
    def encode_image(self, image):

        # process image with CLIP
        image_features = self.clip_model.encode_image(image)
        return image_features
    
    def encode_title(self, titles):

        # process title with CLIP's text encoder
        text_tokens = clip.tokenize(titles).to(next(self.clip_model.parameters()).device)
        title_features = self.clip_model.encode_text(text_tokens)
        return title_features
    
    def encode_description(self, descriptions):

        # process longer description with DistilBERT
        inputs = self.text_tokenizer(descriptions, padding=True, truncation=True, 
                                     return_tensors="pt").to(next(self.text_model.parameters()).device)
        outputs = self.text_model(**inputs)
            
        # use mean pooling for the description (another option is to use CLS token but less stable in practice)
        desc_features = outputs.last_hidden_state.mean(dim=1)
        
        # project to match CLIP dimensions
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