import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch

class BertEncoder(nn.Module):
    def __init__(self, output_dim=256, dropout=0.3, frozen=True, semifrozen=False):
        super().__init__()
        self.titlemodel = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.title_dim = 768

        if frozen:
            for param in self.titlemodel.parameters():
                param.requires_grad = False
        
        elif semifrozen:
            for name, param in self.titlemodel.named_parameters():
                if name.startswith("encoder.layer.11") or name.startswith("pooler"):
                    param.requires_grad = True
                    print(f"Unfreezing {name}")
                else:
                    param.requires_grad = False
        
        # Projection layer to reduce dimensions
        self.title_projector = nn.Sequential(
            nn.Linear(self.title_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, descriptions):
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
        output = self.titlemodel(**encoded_input)
        
        desc_features = output.last_hidden_state.mean(dim=1)
        desc_features = self.title_projector(desc_features)
        return desc_features