import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class HallucinationPropagation:
    """
    Implementation of hallucination propagation using attention weights (based on Section 3.2 of paper).
    
    Key idea: 
    If a token attends strongly to previous unreliable tokens, it becomes
    less reliable itself. This addresses the "overconfidence problem" where models
    assign high probabilities to hallucinated tokens that are consistent with 
    previous hallucinated context.
    
    Example from paper: "2012 Summer Olympics" gets high probability because it
    attends to the earlier (hallucinated) mention of "2012".
    """
    
    def __init__(self, gamma: float = 0.9):
        """
        Args:
            gamma: Just a decay factor for multi-hop propagation (0 <= gamma <= 1)
                  Higher gamma = penalties propagate further
        """
        self.gamma = gamma
    
    def extract_attention_weights(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights from the model.
        
        Args:
            model: The language model
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            attention_weights: Averaged attention weights [batch_size, seq_len, seq_len]
                              attention_weights[b,i,j] = how much token i attends to token j
        """
        model.eval()
        with torch.no_grad():
            # For HookedTransformer, we can get attention through run_with_cache
            _, cache = model.run_with_cache(input_ids, attention_mask=attention_mask)
            
            # Extract attention patterns from all layers and heads
            # Observation_ The paper uses max-pooling across layers and heads
            attention_patterns = []
            
            for layer_idx in range(model.cfg.n_layers):
                layer_attn = cache[f'blocks.{layer_idx}.attn.hook_pattern']  # [batch, heads, seq, seq]
                attention_patterns.append(layer_attn)
            
            # Stack all layers: [layers, batch, heads, seq, seq]
            all_attention = torch.stack(attention_patterns, dim=0)
            
            # Max-pool across layers and heads as mentioned in the paper
            # First max-pool across heads, then across layers
            max_across_heads = torch.max(all_attention, dim=2)[0]  # [layers, batch, seq, seq]
            max_across_layers = torch.max(max_across_heads, dim=0)[0]  # [batch, seq, seq]
            
            return max_across_layers
    
    def compute_propagation_penalties(self, 
                                    hallucination_scores: torch.Tensor,
                                    attention_weights: torch.Tensor,
                                    keyword_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute propagation penalties using attention weights and previous hallucination scores.
        

        This implements Equations 4, 5, and 6 from the paper:
        
        ĥ_i = h_i + I(t_i ∈ K) * γ * p_i
        p_i = ∑_{j=0}^{i-1} w_{i,j} * ĥ_j
        w_{i,j} = I(t_i ∈ K) * att_{i,j} / ∑_{k=0}^{i-1} I(t_i ∈ K) * att_{i,k}
        
        Args:
            hallucination_scores: Initial h_i scores [batch_size, seq_len]
            attention_weights: Attention matrix [batch_size, seq_len, seq_len]
            keyword_mask: Binary mask for keywords [batch_size, seq_len]
            
        Returns:
            updated_scores: ĥ_i scores with propagation penalties [batch_size, seq_len]
        """
        batch_size, seq_len = hallucination_scores.shape
        updated_scores = hallucination_scores.clone()
        
        # Process each position sequentially (can't parallelize due to dependency)
        for i in range(1, seq_len):  # Start from 1 since position 0 has no previous tokens
            if not keyword_mask[:, i].any():  # Skip if not a keyword at any batch position
                continue
            
            # Get attention weights from current position to all previous positions
            curr_attention = attention_weights[:, i, :i]  # [batch_size, i]
            
            # Only consider attention to previous keywords (multiply by keyword_mask)
            prev_keyword_mask = keyword_mask[:, :i]  # [batch_size, i]
            masked_attention = curr_attention * prev_keyword_mask.float()
            
            # Normalize attention weights (Equation 6)
            attention_sum = masked_attention.sum(dim=1, keepdim=True)  # [batch_size, 1]
            # Avoid division by zero
            attention_sum = torch.where(attention_sum > 0, attention_sum, torch.ones_like(attention_sum))
            normalized_attention = masked_attention / attention_sum  # [batch_size, i]
            
            # Compute penalty as weighted sum of previous updated scores (Equation 5)
            prev_scores = updated_scores[:, :i]  # [batch_size, i]
            penalty = torch.sum(normalized_attention * prev_scores, dim=1)  # [batch_size]
            
            # Apply penalty only to keywords (Equation 4)
            keyword_indicator = keyword_mask[:, i].float()  # [batch_size]
            penalty = keyword_indicator * self.gamma * penalty
            
            # Update scores
            updated_scores[:, i] = updated_scores[:, i] + penalty
        
        return updated_scores
    
    def apply_propagation(self, 
                         model,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor, 
                         hallucination_scores: torch.Tensor,
                         keyword_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply hallucination propagation to uncertainty scores.
        
        Args:
            model: Language model
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            hallucination_scores: Initial uncertainty scores [batch_size, seq_len]
            keyword_mask: Binary mask indicating keywords [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
            - 'propagated_scores': Updated hallucination scores
            - 'penalties': Just the penalty values added
            - 'attention_weights': The attention matrix used
        """
        # Extract attention weights
        attention_weights = self.extract_attention_weights(model, input_ids, attention_mask)
        
        # Compute propagated scores
        propagated_scores = self.compute_propagation_penalties(
            hallucination_scores, attention_weights, keyword_mask
        )
        
        # Calculate just the penalties for analysis
        penalties = propagated_scores - hallucination_scores
        
        return {
            'propagated_scores': propagated_scores,
            'penalties': penalties,
            'attention_weights': attention_weights
        }