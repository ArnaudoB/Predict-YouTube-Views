import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

class QwenEncoder(nn.Module):
    def __init__(self, output_dim=256, dropout=0.2, frozen=True, semifrozen=False, layers_to_unfreeze=2):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Quantization config
        self.quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Load model in 8-bit quantized mode
        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen3-0.6B",
            quantization_config=self.quant_config,
            device_map="auto"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B"
        )

        self.title_dim = self.model.config.hidden_size
        self.layers_to_unfreeze = layers_to_unfreeze

        # Freezing or semi-freezing layers (unused because of quantization)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        elif semifrozen:
            for name, param in self.model.named_parameters():
                if param.dtype not in [torch.float, torch.float16, torch.bfloat16, torch.complex64, torch.complex128]:
                    continue  # skip non-trainable dtypes
                if any(f"layers.{i}" in name for i in range(self.model.config.num_hidden_layers - self.layers_to_unfreeze, self.model.config.num_hidden_layers)):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Projection head
        self.title_projector = nn.Sequential(
            nn.Linear(self.title_dim, output_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        ).to(self.device)

        self._init_weights()

    def _init_weights(self):
        for layer in self.title_projector:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=0.1, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, titles):

        titles = [str(t) if t is not None else "" for t in titles]
        
        encoded_input = self.tokenizer(
            titles,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        outputs = self.model(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            output_hidden_states=True
        )

        last_hidden_state = outputs.hidden_states[-2]  # (batch_size, seq_len, hidden_dim)

        last_hidden_state = self.model.norm(last_hidden_state)

        attention_mask = encoded_input["attention_mask"].unsqueeze(-1)
        masked_hidden = last_hidden_state * attention_mask
        sum_embeddings = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask

        projected = self.title_projector(pooled)
        return projected