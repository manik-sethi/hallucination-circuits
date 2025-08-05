import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import spacy

from TokenUnertaintyComputer import TokenUnertaintyComputer
from HallucinationPropagation import HallucinationPropagation  
from ProbabilityCorrection import ProbabilityCorrection

class FOCUSHallucinationDetector:
    """
    Main class implementing the FOCUS method for hallucination detection.
    
    Combines all three focus mechanisms:
    1. Focus on informative keywords (named entities & nouns)
    2. Focus on preceding words (hallucination propagation via attention)
    3. Focus on token properties (entity type prompting + IDF correction)
    
    This is the complete implementation of the method from:
    "Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus"
    """
    
    def __init__(self, 
                 model,
                 tokenizer, 
                 device: str = "cpu",
                 gamma: float = 0.9,
                 rho: float = 0.01,
                 idf_corpus_file: Optional[str] = None):
        """
        Args:
            model: Language model (HookedTransformer or similar)
            tokenizer: Model's tokenizer
            device: Device for computations
            gamma: Decay factor for hallucination propagation (0 <= gamma <= 1)
            rho: Threshold for entity type candidate selection
            idf_corpus_file: Path to corpus for IDF computation (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma
        self.rho = rho
        
        # Initialize the three components
        self.uncertainty_computer = TokenUnertaintyComputer(model, tokenizer, device)
        self.propagation = HallucinationPropagation(gamma)
        self.prob_correction = ProbabilityCorrection(model, tokenizer, idf_corpus_file)
        
        # Share spaCy instance across components
        self.nlp = self.uncertainty_computer.nlp
        self.prob_correction.nlp = self.nlp
    
    def detect_hallucinations(self, 
                            text: str,
                            return_details: bool = False,
                            apply_keyword_focus: bool = True,
                            apply_propagation: bool = True, 
                            apply_prob_correction: bool = True) -> Dict[str, Union[float, torch.Tensor, Dict]]:
        """
        Main method to detect hallucinations in text using the FOCUS approach.
        
        Args:
            text: Input text to analyze
            return_details: Whether to return detailed intermediate results
            apply_keyword_focus: Whether to apply keyword focusing (Focus #1)
            apply_propagation: Whether to apply hallucination propagation (Focus #2)
            apply_prob_correction: Whether to apply probability correction (Focus #3)
            
        Returns:
            Dictionary containing:
            - 'sentence_score': Overall hallucination score for the text
            - 'token_scores': Per-token hallucination scores [seq_len]
            - 'keywords': List of identified keywords
            - 'details': Detailed intermediate results (if return_details=True)
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # ============================================================================
        # STEP 1: KEYWORD IDENTIFICATION (Using TokenUnertaintyComputer)
        # ============================================================================
        print("🔍 Step 1: Identifying keywords...")
        keywords = self.uncertainty_computer.identify_keywords(text)
        token_mapping = self.uncertainty_computer.map_text_to_model_tokens(text, input_ids)
        
        # Create keyword mask for model tokens
        keyword_mask = self._create_keyword_mask(input_ids, keywords, token_mapping)
        print(f"   Found {len(keywords)} keywords: {[kw[2] for kw in keywords]}")
        
        # ============================================================================
        # STEP 2: UNCERTAINTY COMPUTATION (Using TokenUnertaintyComputer OR ProbabilityCorrection)
        # ============================================================================
        if apply_prob_correction:
            print(" Step 2: Computing uncertainty with probability correction...")
            # Use ProbabilityCorrection class for corrected probabilities
            uncertainty_results = self.prob_correction.recompute_uncertainty_with_correction(
                input_ids, attention_mask,
                entity_type_prompt=True,
                apply_idf=bool(self.prob_correction.idf_scores)
            )
            hallucination_scores = uncertainty_results['corrected_hallucination_score']
            print(f"   Applied entity type prompting: True")
            print(f"   Applied IDF correction: {bool(self.prob_correction.idf_scores)}")
        else:
            print("📊 Step 2: Computing basic uncertainty (no correction)...")
            # Use TokenUnertaintyComputer for basic uncertainty
            uncertainty_results = self.uncertainty_computer.compute_basic_uncertainty(input_ids, attention_mask)
            hallucination_scores = uncertainty_results['hallucination_score']
        
        print(f"   Initial hallucination scores computed: mean = {hallucination_scores.mean():.4f}")
        
        # ============================================================================
        # STEP 3: HALLUCINATION PROPAGATION (Using HallucinationPropagation)
        # ============================================================================
        if apply_propagation:
            print("🔗 Step 3: Applying hallucination propagation...")
            propagation_results = self.propagation.apply_propagation(
                self.model, input_ids, attention_mask, hallucination_scores, keyword_mask
            )
            hallucination_scores = propagation_results['propagated_scores']
            
            # Show how much scores changed due to propagation
            penalties = propagation_results['penalties']
            avg_penalty = penalties[keyword_mask].mean() if keyword_mask.any() else 0
            print(f"   Applied attention-based propagation (γ={self.gamma})")
            print(f"   Average penalty to keywords: {avg_penalty:.4f}")
        else:
            print(" Step 3: Skipping hallucination propagation...")
            propagation_results = None
        
        # ============================================================================
        # STEP 4: SENTENCE-LEVEL SCORING (Keyword Focus)
        # ============================================================================
        if apply_keyword_focus:
            print("🎯 Step 4: Computing sentence score (keyword focus only)...")
            # Focus only on keywords (Equation 3 from paper)
            sentence_score = self._compute_keyword_focused_sentence_score(
                hallucination_scores, keyword_mask
            )
            print(f"   Sentence score (keywords only): {sentence_score:.4f}")
        else:
            print(" Step 4: Computing sentence score (all tokens)...")
            # Use all tokens
            sentence_score = hallucination_scores.mean().item()
            print(f"   Sentence score (all tokens): {sentence_score:.4f}")
        
        # Prepare results
        results = {
            'sentence_score': sentence_score,
            'token_scores': hallucination_scores.squeeze(0).cpu(),  # Remove batch dimension
            'keywords': keywords,
            'keyword_mask': keyword_mask.squeeze(0).cpu(),
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        }
        
        if return_details:
            results['details'] = {
                'uncertainty_results': uncertainty_results,
                'propagation_results': propagation_results,
                'token_mapping': token_mapping,
                'input_ids': input_ids.squeeze(0).cpu(),
                'attention_mask': attention_mask.squeeze(0).cpu()
            }
        
        return results
    
    def _create_keyword_mask(self, 
                           input_ids: torch.Tensor,
                           keywords: List[Tuple[int, int, str]], 
                           token_mapping: Dict[int, List[int]]) -> torch.Tensor:
        """
        Create binary mask indicating which model tokens are keywords.
        
        Args:
            input_ids: Model token IDs [batch_size, seq_len]
            keywords: List of (start, end, type) from spaCy
            token_mapping: Mapping from spaCy tokens to model tokens
            
        Returns:
            keyword_mask: Binary tensor [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        keyword_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        # Mark keyword positions
        for start_spacy, end_spacy, _ in keywords:
            for spacy_idx in range(start_spacy, end_spacy):
                if spacy_idx in token_mapping:
                    for model_idx in token_mapping[spacy_idx]:
                        if model_idx < seq_len:
                            keyword_mask[:, model_idx] = True
        
        return keyword_mask
    
    def _compute_keyword_focused_sentence_score(self, 
                                              hallucination_scores: torch.Tensor,
                                              keyword_mask: torch.Tensor) -> float:
        """
        Compute sentence-level hallucination score focusing only on keywords.
        
        Implements Equation 3 from the paper:
        h^s = (1 / Σ I(t_i ∈ K)) * Σ I(t_i ∈ K) * h_i
        
        Args:
            hallucination_scores: Token-level scores [batch_size, seq_len]
            keyword_mask: Binary mask for keywords [batch_size, seq_len]
            
        Returns:
            Sentence-level hallucination score
        """
        # Mask scores to only include keywords
        keyword_scores = hallucination_scores * keyword_mask.float()
        
        # Compute weighted average
        total_keyword_score = keyword_scores.sum()
        num_keywords = keyword_mask.sum().float()
        
        if num_keywords > 0:
            sentence_score = (total_keyword_score / num_keywords).item()
        else:
            # Fallback to all tokens if no keywords found
            sentence_score = hallucination_scores.mean().item()
        
        return sentence_score
    
    def analyze_batch(self, 
                     texts: List[str],
                     **kwargs) -> List[Dict]:
        """
        Analyze a batch of texts for hallucinations.
        
        Args:
            texts: List of input texts
            **kwargs: Arguments passed to detect_hallucinations()
            
        Returns:
            List of result dictionaries, one per text
        """
        results = []
        for text in texts:
            result = self.detect_hallucinations(text, **kwargs)
            results.append(result)
        return results
    
    def get_top_hallucinated_tokens(self, 
                                  result: Dict,
                                  top_k: int = 10) -> List[Tuple[str, float, bool]]:
        """
        Get the top-k most likely hallucinated tokens from a result.
        
        Args:
            result: Result from detect_hallucinations()
            top_k: Number of top tokens to return
            
        Returns:
            List of (token, score, is_keyword) tuples sorted by score
        """
        token_scores = result['token_scores']
        tokens = result['tokens']
        keyword_mask = result['keyword_mask']
        
        # Create list of (token, score, is_keyword)
        token_info = []
        for i, (token, score, is_keyword) in enumerate(zip(tokens, token_scores, keyword_mask)):
            token_info.append((token, float(score), bool(is_keyword)))
        
        # Sort by score (descending) and return top-k
        token_info.sort(key=lambda x: x[1], reverse=True)
        return token_info[:top_k]
    
    def compare_methods(self, text: str) -> Dict[str, float]:
        """
        Compare hallucination scores with different combinations of focus mechanisms.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with scores from different method combinations
        """
        methods = {
            'baseline': {'apply_keyword_focus': False, 'apply_propagation': False, 'apply_prob_correction': False},
            'keywords_only': {'apply_keyword_focus': True, 'apply_propagation': False, 'apply_prob_correction': False},
            'propagation_only': {'apply_keyword_focus': False, 'apply_propagation': True, 'apply_prob_correction': False},
            'correction_only': {'apply_keyword_focus': False, 'apply_propagation': False, 'apply_prob_correction': True},
            'keywords_propagation': {'apply_keyword_focus': True, 'apply_propagation': True, 'apply_prob_correction': False},
            'keywords_correction': {'apply_keyword_focus': True, 'apply_propagation': False, 'apply_prob_correction': True},
            'propagation_correction': {'apply_keyword_focus': False, 'apply_propagation': True, 'apply_prob_correction': True},
            'full_focus': {'apply_keyword_focus': True, 'apply_propagation': True, 'apply_prob_correction': True}
        }
        
        scores = {}
        for method_name, params in methods.items():
            result = self.detect_hallucinations(text, return_details=False, **params)
            scores[method_name] = result['sentence_score']
        
        return scores