"""
Model implementations for AMR entailment task.
"""
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import (
    RobertaModel, 
    RobertaPreTrainedModel, 
    RobertaConfig,
    PreTrainedModel
)


class AMREntailmentRoBERTa(RobertaPreTrainedModel):
    """
    RoBERTa model customized for the AMR entailment task.
    
    This model uses a RoBERTa base with a regression head to predict
    entailment scores (1.0 for entailment, 0.0 for non-entailment).
    """
    
    def __init__(self, config: RobertaConfig):
        """
        Initialize the model with a RoBERTa base and classification head.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()  # Sigmoid to output values between 0 and 1
        )
        
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask for padding
            token_type_ids: Token type IDs
            position_ids: Position IDs
            head_mask: Head mask
            labels: Binary labels (1.0=entails, 0.0=not entails)
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Dict with loss and logits
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the [CLS] token representation (sentence representation)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Pass through the classifier
        logits = self.classifier(cls_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(logits.view(-1), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
            "attentions": outputs.attentions if output_attentions else None,
        }


def load_pretrained_model(
    model_name_or_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[PreTrainedModel, RobertaConfig]:
    """
    Load a pretrained model for AMR entailment.
    
    Args:
        model_name_or_path: Either a HuggingFace model name or a path to a saved model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config)
    """
    # Load configuration
    config = RobertaConfig.from_pretrained(model_name_or_path)
    
    # Create model
    model = AMREntailmentRoBERTa.from_pretrained(
        model_name_or_path,
        config=config
    )
    
    # Move model to device
    model = model.to(device)
    
    return model, config
