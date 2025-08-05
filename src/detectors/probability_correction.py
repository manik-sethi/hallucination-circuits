import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math

class ProbabilityCorrection:
    """
    Implements probability correction to address the "underconfidence problem" (Section 3.3).
    
    Two main issues this solves:
    1. Models may assign low probability to factual tokens when many plausible options exist
    2. Rare tokens get systematically lower probabilities regardless of factuality
    
    Solutions:
    1. Entity type prompting: Insert entity types before named entities to constrain generation
    2. IDF weighting: Adjust probabilities based on token frequency
    """
    
    def __init__(self, model, tokenizer, idf_corpus_file: Optional[str] = None):
        """
        Args:
            model: The language model
            tokenizer: Model's tokenizer  
            idf_corpus_file: Path to corpus for computing IDF scores (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.idf_scores = {}
        
        if idf_corpus_file:
            self._compute_idf_scores(idf_corpus_file)
    
    def _compute_idf_scores(self, corpus_file: str):
        """
        Compute IDF scores from a corpus file.
        IDF(token) = log(N / df(token)) where N = total docs, df = doc frequency
        
        Args:
            corpus_file: Path to text corpus (one document per line)
        """
        print(f"Computing IDF scores from {corpus_file}...")
        
        # Count document frequency for each token
        doc_count = 0
        token_doc_freq = Counter()
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc_count += 1
                doc_text = line.strip()
                
                # Tokenize and get unique tokens for this document
                tokens = self.tokenizer.encode(doc_text)
                unique_tokens = set(tokens)
                
                for token in unique_tokens:
                    token_doc_freq[token] += 1
                
                if doc_count % 10000 == 0:
                    print(f"Processed {doc_count} documents...")
        
        # Compute IDF scores
        for token, freq in token_doc_freq.items():
            self.idf_scores[token] = math.log(doc_count / freq)
        
        print(f"Computed IDF for {len(self.idf_scores)} tokens from {doc_count} documents")
    
    def get_token_idf(self, token_id: int) -> float:
        """Get IDF score for a token (default to 1.0 if not in corpus)."""
        return self.idf_scores.get(token_id, 1.0)
    
    def create_entity_type_prompt(self, text: str, keywords: List[Tuple[int, int, str]]) -> str:
        """
        Insert entity type information before named entities in the text.
        
        This implements the entity type prompting from Section 3.3.
        Example: "Born in 1992" -> "Born in <DATE> 1992"
        
        Args:
            text: Original text
            keywords: List of (start, end, entity_type) from spaCy
            
        Returns:
            Modified text with entity type tags
        """
        # Sort keywords by position (reversed to avoid position shifting)
        sorted_keywords = sorted(keywords, key=lambda x: x[0], reverse=True)
        
        doc = self.nlp(text) if hasattr(self, 'nlp') else None
        if doc is None:
            return text  # Fallback if spaCy not available
        
        modified_text = text
        tokens = list(doc)
        
        for start_idx, end_idx, entity_type in sorted_keywords:
            if entity_type == 'NOUN':  # Skip regular nouns, only tag named entities
                continue
                
            if start_idx < len(tokens) and end_idx <= len(tokens):
                # Get character positions
                start_char = tokens[start_idx].idx
                end_char = tokens[end_idx - 1].idx + len(tokens[end_idx - 1].text)
                
                # Insert entity type tag
                original_entity = modified_text[start_char:end_char]
                tagged_entity = f"<{entity_type}> {original_entity}"
                
                modified_text = modified_text[:start_char] + tagged_entity + modified_text[end_char:]
        
        return modified_text
    
    def compute_corrected_probabilities(self, 
                                      input_ids: torch.Tensor,
                                      attention_mask: torch.Tensor,
                                      entity_type_prompt: bool = True,
                                      apply_idf: bool = True,
                                      rho: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Compute probability corrections using entity type prompting and IDF weighting.
        
        This implements Equations 7 and 8 from the paper.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]  
            entity_type_prompt: Whether to use entity type prompting
            apply_idf: Whether to apply IDF correction
            rho: Threshold for candidate set in entity type prompting
            
        Returns:
            Dictionary with corrected probabilities and intermediate results
        """
        # Get original probabilities
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)
            original_probs = F.softmax(logits, dim=-1)
        
        corrected_probs = original_probs.clone()
        
        if entity_type_prompt:
            # Entity type correction approximation
            # In practice, this would require re-running the model with entity type prompts
            # For now, we'll simulate by boosting probabilities of tokens that match expected types
            corrected_probs = self._apply_entity_type_correction(corrected_probs, input_ids, rho)
        
        if apply_idf and self.idf_scores:
            # Apply IDF weighting (Equation 8)
            corrected_probs = self._apply_idf_correction(corrected_probs, input_ids)
        
        return {
            'original_probs': original_probs,
            'corrected_probs': corrected_probs,
            'correction_factor': corrected_probs / (original_probs + 1e-10)
        }
    
    def _apply_entity_type_correction(self, probs: torch.Tensor, input_ids: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Apply entity type correction by renormalizing among likely candidates.
        
        This is a simplified version of the entity type prompting.
        In the full implementation, you would re-run the model with entity type hints.
        """
        batch_size, seq_len, vocab_size = probs.shape
        corrected_probs = probs.clone()
        
        for b in range(batch_size):
            for i in range(seq_len):
                # Get top candidates above threshold
                position_probs = probs[b, i]
                top_candidates = position_probs > rho
                
                if top_candidates.sum() > 1:  # Only renormalize if multiple candidates
                    # Renormalize among candidates (approximates Equation 7)
                    candidate_probs = position_probs * top_candidates.float()
                    normalization = candidate_probs.sum()
                    
                    if normalization > 0:
                        corrected_probs[b, i] = candidate_probs / normalization
        
        return corrected_probs
    
    def _apply_idf_correction(self, probs: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply IDF correction to boost probabilities of rare tokens.
        
        Implements Equation 8: p̂(t) = p̃(t) * idf(t) / Σ_v p̃(v) * idf(v)
        """
        batch_size, seq_len, vocab_size = probs.shape
        
        # Create IDF weight tensor for the entire vocabulary
        idf_weights = torch.ones(vocab_size, device=probs.device)
        for token_id, idf_score in self.idf_scores.items():
            if token_id < vocab_size:
                idf_weights[token_id] = idf_score
        
        # Apply IDF weighting
        weighted_probs = probs * idf_weights.unsqueeze(0).unsqueeze(0)  # Broadcast to [batch, seq, vocab]
        
        # Renormalize
        normalization = weighted_probs.sum(dim=-1, keepdim=True)
        corrected_probs = weighted_probs / (normalization + 1e-10)
        
        return corrected_probs
    
    def recompute_uncertainty_with_correction(self,
                                            input_ids: torch.Tensor,
                                            attention_mask: torch.Tensor,
                                            entity_type_prompt: bool = True,
                                            apply_idf: bool = True) -> Dict[str, torch.Tensor]:
        """
        Recompute uncertainty scores using corrected probabilities.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            entity_type_prompt: Whether to use entity type correction
            apply_idf: Whether to apply IDF correction
            
        Returns:
            Dictionary with corrected uncertainty scores
        """
        # Get corrected probabilities
        prob_results = self.compute_corrected_probabilities(
            input_ids, attention_mask, entity_type_prompt, apply_idf
        )
        corrected_probs = prob_results['corrected_probs']
        
        # Get probabilities of actual tokens
        actual_token_probs = torch.gather(corrected_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        
        # Recompute uncertainty with corrected probabilities
        neg_log_prob = -torch.log(actual_token_probs + 1e-10)
        entropy = -torch.sum(corrected_probs * torch.log(corrected_probs + 1e-10), dim=-1)
        corrected_hallucination_score = neg_log_prob + entropy
        
        return {
            'corrected_neg_log_prob': neg_log_prob,
            'corrected_entropy': entropy,
            'corrected_hallucination_score': corrected_hallucination_score,
            'corrected_token_probs': actual_token_probs,
            **prob_results
        }