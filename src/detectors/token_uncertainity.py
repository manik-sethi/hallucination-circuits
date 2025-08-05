import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import spacy
from collections import defaultdict

class TokenUnertaintyComputer:
    """
    Computes token-level uncertainty scores for hallucination detection,
    based on the paper "Hallucination Detection in Language Models via Token Uncertainty".
    
    This is the base class that handles:
    1. Getting token probabilities from the model
    2. Computing uncertainty metrics (negative log probability + entropy, as equation in the paper)
    3. Identifying keywords (named entities and nouns) ( spaCy)
    
    The core idea: 
    tokens that are hallucinated should have lower probabilities
    according to a well-trained model, since they deviate from learned patterns.
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        """
        Args:
            model: 
            tokenizer: Model's tokenizer
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(" install spacy English model: python -m spacy download en_core_web_sm")
            raise
    
    def get_token_probabilities(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token probabilities and logits from the model.
        
        Args:

            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:

            probs: Token probabilities [batch_size, seq_len, vocab_size]
            logits: Raw logits [batch_size, seq_len, vocab_size]
        """


        self.model.eval()
        with torch.no_grad():
            # Get model outputs - this works for HookedTransformer
            logits = self.model(input_ids, attention_mask=attention_mask)
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
        return probs, logits
    
    def compute_basic_uncertainty(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute basic uncertainty metrics for each token.
        
        This implements Equations 1 and 2 from the paper:
        h_i = -log(p_i(t_i)) + H_i 
        --> the local and global uncertainity scores for token t_i

        where:
        H_i = -∑_v p_i(v) * log2(p_i(v)) ;; p_i(v) is the probability of token v at position i
        
        Args:

            input_ids: Token IDs [batch_size, seq_len] 
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
            - 'neg_log_prob': Negative log probability of actual tokens
            - 'entropy': Entropy at each position
            - 'hallucination_score': Combined uncertainty score (neg_log_prob + entropy) --> not the final score
            - 'token_probs': Probabilities of the actual tokens at each position 
        """

        probs, logits = self.get_token_probabilities(input_ids, attention_mask)
        batch_size, seq_len, vocab_size = probs.shape
        
        # Get probabilities of the actual tokens
        # input_ids[i,j] gives the actual token at position (i,j)
        # We want probs[i,j,input_ids[i,j]] - the probability assigned to that token
        actual_token_probs = torch.gather(probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        
        # Negative log probability of actual tokens
        neg_log_prob = -torch.log(actual_token_probs + 1e-10)  # Added small epsilon for numerical stability
        
        # Entropy: H_i = -∑_v p_i(v) * log2(p_i(v))
        # We use natural log instead of log2 for consistency
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Combined hallucination score (Equation 1)
        hallucination_score = neg_log_prob + entropy
        
        return {
            'neg_log_prob': neg_log_prob,
            'entropy': entropy, 
            'hallucination_score': hallucination_score,
            'token_probs': actual_token_probs
        }
    


    def identify_keywords(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Identify keywords (named entities and nouns) in the text using spaCy.
        
        This implements the keyword selection from Section 3.1 of the paper.
        The paper focuses on:
        1. 18 types of named entities (PERSON, ORG, DATE, etc.)
        2. Nouns that aren't named entities
        
        Args:
            text: Input text string
            
        Returns:
            List of (start_token_idx, end_token_idx, entity_type) tuples
        """
        doc = self.nlp(text)
        keywords = []
        
        # Get named entities
        for ent in doc.ents:
            # Convert character positions to token positions
            start_token = None
            end_token = None
            
            for i, token in enumerate(doc):
                if token.idx == ent.start_char:
                    start_token = i
                if token.idx + len(token.text) == ent.end_char:
                    end_token = i + 1
                    break
            
            if start_token is not None and end_token is not None:
                keywords.append((start_token, end_token, ent.label_))
        
        # Get nouns that aren't already named entities
        entity_tokens = set()
        for start, end, _ in keywords:
            entity_tokens.update(range(start, end))
        
        for i, token in enumerate(doc):
            if token.pos_ == 'NOUN' and i not in entity_tokens:
                keywords.append((i, i+1, 'NOUN'))
        
        return keywords
    
    def map_text_to_model_tokens(self, text: str, input_ids: torch.Tensor) -> Dict[int, List[int]]:
        """
        Map spaCy token positions to model token positions.
        
        This is tricky because spaCy tokenization != model tokenization.
        I aligned them to know which model tokens correspond to keywords.
        
        Args:
            text: Original text
            input_ids: Model token IDs
            
        Returns:
            Dictionary mapping spacy_token_idx -> [model_token_indices]
        """
        # Decode model tokens to get text spans
        model_tokens = []
        for i in range(input_ids.shape[-1]):
            token_text = self.tokenizer.decode([input_ids[0, i]], skip_special_tokens=True)
            model_tokens.append(token_text)
        
        # Simple alignment - this could be improved with more sophisticated methods
        spacy_doc = self.nlp(text)
        spacy_to_model = defaultdict(list)
        
        model_idx = 0
        for spacy_idx, spacy_token in enumerate(spacy_doc):
            # Try to find matching model tokens
            spacy_text = spacy_token.text.lower().strip()
            
            while model_idx < len(model_tokens):
                model_text = model_tokens[model_idx].lower().strip()
                
                if spacy_text in model_text or model_text in spacy_text:
                    spacy_to_model[spacy_idx].append(model_idx)
                    model_idx += 1
                    break
                elif model_text == '':  # Skip empty tokens
                    model_idx += 1
                else:
                    model_idx += 1  
                    break
        
        return spacy_to_model