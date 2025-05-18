import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel

class ClipBertEncoder(nn.Module):
    def __init__(self, frozen=True, semifrozen=False):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            for param in self.img_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.descmodel.parameters():
                param.requires_grad = False
        
        # Unfreeze the last layer of every model :
        elif semifrozen:
            # for BERT
            for name, param in self.descmodel.named_parameters():
                if name.startswith("encoder.layer.11") or name.startswith("pooler"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # for CLIP
            for name, param in self.img_model.vision_model.named_parameters():
                if "encoder.layers.11" in name:  # dernière couche ViT
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # for SentenceTransformer
            for name, param in self.text_model.named_parameters():
                if "transformer.layer.11" in name:  # pour DistilBERT/BERT-like
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Project description embeddings to match CLIP dimensions
        self.desc_projector = nn.Linear(self.desc_dim, self.img_dim)
        
        # Fusion layer to combine all three embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.img_dim * 3, self.img_dim),
            nn.ReLU()
        )
        
    def encode_image(self, images):
        inputs = self.img_processor(images=images, return_tensors="pt", do_rescale=False) # processes the images (already scaled in dataset)
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.img_model.device) for k, v in inputs.items()}
        # Get image features

        img_feat = self.img_model.get_image_features(**inputs)
        return img_feat
    
    def encode_title(self, titles):
        encoded_titles = self.text_model.encode(titles) # returns a numpy array by default
        return torch.tensor(encoded_titles).float().to(self.device)
    
    def encode_description(self, descriptions):
    # Convert None values to empty strings and handle list of descriptions
        descriptions = [str(desc) if desc is not None else "" for desc in descriptions]
        
        # Add truncation and padding parameters
        encoded_input = self.tokenizer(
            descriptions, 
            return_tensors='pt', 
            padding=True,       # Add padding to make all sequences the same length
            truncation=True,    # Truncate sequences that are too long
            max_length=512      # Specify maximum length (BERT's limit is 512)
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
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