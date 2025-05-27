# Implementation of SynthID-Text watermarking (Vectorized Binary N=2)
# Based on Corollary 14 in https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf

from typing import List, Optional

import math
import torch
import random
import numpy as np

from torch import Tensor
from transformers import LogitsProcessor
from watermark.old_watermark import top_p_indices
import pdb  # For debugging purposes, can be removed in production
from scipy.stats import norm

class SynthIDLogitsProcessor(LogitsProcessor):


    def __init__(self,
                 vocab_size: int,
                 hash_key: int = 15485863, # Default from Kirchenbauer et al.
                 device: torch.device = None,
                 dynamic_seed: str = "markov_1", # "initial", None
                 initial_seed: int = None,
                 store_g_values: bool = False,
                 top_p: float = 0.999,
                 temperature=1.0
                 ):

        self.vocab_size = vocab_size
        self.hash_key = hash_key
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dynamic_seed = dynamic_seed
        self.initial_seed = initial_seed
        self.store_g_values = store_g_values
        self.g_values_history = None # To store g-values if store_g_values is True
        self.top_p = top_p
        self.seed_increment = 0 # For fresh randomness if needed
        self.saved_distributions = []
        self.temperature = temperature
        self.ts_K = 15


        if dynamic_seed == "initial" or dynamic_seed is None:
            assert initial_seed is not None, "initial_seed must be provided if dynamic_seed is 'initial' or None."

        # Precompute vocabulary indices tensor on the correct device
        self.vocab_indices = torch.arange(self.vocab_size, device=self.device)

        # Initialize RNG
        self.rng = torch.Generator(device=self.device)
        if initial_seed is not None and dynamic_seed is None:
            self.rng.manual_seed(self.hash_key * self.initial_seed)


    def _get_seed_for_token(self, input_ids: torch.LongTensor, batch_idx: int) -> int:
        """Calculates the seed for the current token based on the dynamic_seed setting."""
        if self.dynamic_seed == "markov_1":
            if input_ids.shape[1] == 1: # Handle the very first token generation
                 # Use a default seed or the initial_seed if available for the first token
                 return self.hash_key * (self.initial_seed if self.initial_seed is not None else 0)
            else:
                prev_token = input_ids[batch_idx, -1].item()
                return self.hash_key * prev_token
        elif self.dynamic_seed == "initial":
            return self.hash_key * self.initial_seed
        elif self.dynamic_seed is None:
             # Use the fixed RNG state (seeded once in __init__)
             # Note: This is generally not secure as the sequence is predictable
             return 0 # Seed value doesn't matter here as we use the generator state
        elif self.dynamic_seed == "fresh":
            # For fresh randomness, we increment the seed based on the current state
            # This is a simple way to ensure each call has a unique seed
            self.seed_increment += 1
            return self.hash_key + self.seed_increment
        else:
            raise ValueError(f"Unknown dynamic_seed type: {self.dynamic_seed}")

    def _get_binary_g_values(self, seed: int) -> torch.Tensor:
        """
        Generates binary g-values (0 or 1) for the entire vocabulary based on the seed.
        A simple hash is used here: hash(token_id + seed) % 2.
        More robust hashing could be used in practice.
        """
        # Ensure seeding is correct for the chosen dynamic_seed strategy
        if self.dynamic_seed is not None:
             self.rng.manual_seed(seed)

        # Simple hash: (token_id * large_prime + seed) % vocab_size, then check LSB
        # Using torch.randint for reproducibility with the generator
        # Generate random integers and take modulo 2 for binary values
        # We generate integers in a large range to simulate hashing
        hashed_values = torch.randint(0, self.vocab_size * 10, (self.vocab_size,),
                                      generator=self.rng, device=self.device)
        g_values = hashed_values % 2
        return g_values # Tensor of size [vocab_size] with values 0 or 1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Applies the SynthID vectorized binary watermark (Corollary 14).

        Args:
            input_ids (`torch.LongTensor`):
                Indices of input sequence tokens in the vocabulary. Shape (batch_size, sequence_length).
            scores (`torch.FloatTensor`):
                Logits of the next token distribution. Shape (batch_size, vocab_size).

        Returns:
            `torch.FloatTensor`: The modified logits. Shape (batch_size, vocab_size).
        """
        # pdb.set_trace()
        batch_size = scores.shape[0]
        new_scores = torch.zeros_like(scores)

        if self.store_g_values:
            if self.g_values_history is None:
                self.g_values_history = [[] for _ in range(batch_size)]

        for b_idx in range(batch_size):
            # apply temperature
            scores[b_idx] = scores[b_idx] / self.temperature
            # Calculate current probabilities from logits
            probs = torch.softmax(scores[b_idx], dim=-1)
            self.saved_distributions.append(probs.detach().cpu().clone())
            # top-p filtering
            filter_indices = top_p_indices(probs, self.top_p)
            ####
            p_new = torch.zeros(size=(self.vocab_size,), device=probs.device)
            p_new[filter_indices] = probs[filter_indices]
            p_new = p_new / p_new.sum()
            ####

            ###
            # add a wrapper for several layers - K layers.
            # wrap ardound this with new seed? increase by 1 - wrap the function
            #####
            # p = TS_tilt(seed)
            # TS_tilt 
            #####


            ###
            # add a wrapper for several layers - K layers.
            # wrap ardound this with new seed? increase by 1 - wrap the function
            #####
            # p = TS_tilt(seed)
            # TS_tilt 
            #####


            current_seed = self._get_seed_for_token(input_ids, b_idx)
            # pdb.set_trace()
            p_wm = p_new
            for _ in range(self.ts_K):
                p_wm = self.TS_tilt(current_seed, p_wm)
                current_seed += 1    
            

            new_scores[b_idx] = torch.log(p_wm + 1e-10)

        return new_scores

        
    def TS_tilt(self,seed,p):
        g1_values = self._get_binary_g_values(seed) # Shape [vocab_size]
        mask_g1_1 = (g1_values == 1)
        p_g1_1 = torch.sum(p[mask_g1_1])

        # 3. Calculate p(V | g1=1) = sum of probabilities of tokens with g1=1
        # Ensure g1_values is boolean or compatible type for indexing/masking
        mask_g1_1 = (g1_values == 1)
        p_g1_1 = torch.sum(p[mask_g1_1])

        # Avoid division by zero or log(0) if p_g1_1 is 0 or 1
        # Clamp p_g1_1 to avoid numerical instability
        epsilon = 1e-10
        p_g1_1 = torch.clamp(p_g1_1, epsilon, 1.0 - epsilon)

        # 4. Calculate factors for modifying probabilities based on Corollary 14
        # Factor for g1=1 tokens: (1 + g1(xt,r) - p(V|g1=1)) = 1 + 1 - p_g1_1 = 2 - p_g1_1
        factor_g1_1 = 2.0 - p_g1_1
        # Factor for g1=0 tokens: (1 + g1(xt,r) - p(V|g1=1)) = 1 + 0 - p_g1_1 = 1 - p_g1_1
        factor_g1_0 = 1.0 - p_g1_1

        # 5. Calculate watermarked probabilities p_wm
        # Initialize p_wm with original probabilities
        p_wm = p.clone()

        # Apply factors based on g-values
        # Ensure mask_g1_1 is boolean
        p_wm[mask_g1_1] *= factor_g1_1
        p_wm[~mask_g1_1] *= factor_g1_0 # Apply factor to g1=0 tokens

        # 6. Renormalize probabilities to ensure they sum to 1
        # This step is crucial as the formula in Corollary 14 might not
        # perfectly preserve the sum due to floating point inaccuracies
        # or the nature of the transformation.
        p_wm_sum = torch.sum(p_wm)
        if p_wm_sum > epsilon: # Avoid division by zero
            p_wm /= p_wm_sum
        else:
            # Handle case where all probabilities become near zero (should be rare)
            # Fallback to uniform distribution or original probs
            p_wm = torch.ones_like(p_wm) / self.vocab_size
        return p_wm
    

        

    def TS_tilt(self,seed,p):
        g1_values = self._get_binary_g_values(seed) # Shape [vocab_size]
        mask_g1_1 = (g1_values == 1)
        p_g1_1 = torch.sum(p[mask_g1_1])

        # 3. Calculate p(V | g1=1) = sum of probabilities of tokens with g1=1
        # Ensure g1_values is boolean or compatible type for indexing/masking
        mask_g1_1 = (g1_values == 1)
        p_g1_1 = torch.sum(p[mask_g1_1])
        # 3. Calculate p(V | g1=1) = sum of probabilities of tokens with g1=1
        # Ensure g1_values is boolean or compatible type for indexing/masking
        mask_g1_1 = (g1_values == 1)
        p_g1_1 = torch.sum(p[mask_g1_1])

        # Avoid division by zero or log(0) if p_g1_1 is 0 or 1
        # Clamp p_g1_1 to avoid numerical instability
        epsilon = 1e-10
        p_g1_1 = torch.clamp(p_g1_1, epsilon, 1.0 - epsilon)
        # Avoid division by zero or log(0) if p_g1_1 is 0 or 1
        # Clamp p_g1_1 to avoid numerical instability
        epsilon = 1e-10
        p_g1_1 = torch.clamp(p_g1_1, epsilon, 1.0 - epsilon)

        # 4. Calculate factors for modifying probabilities based on Corollary 14
        # Factor for g1=1 tokens: (1 + g1(xt,r) - p(V|g1=1)) = 1 + 1 - p_g1_1 = 2 - p_g1_1
        factor_g1_1 = 2.0 - p_g1_1
        # Factor for g1=0 tokens: (1 + g1(xt,r) - p(V|g1=1)) = 1 + 0 - p_g1_1 = 1 - p_g1_1
        factor_g1_0 = 1.0 - p_g1_1
        # 4. Calculate factors for modifying probabilities based on Corollary 14
        # Factor for g1=1 tokens: (1 + g1(xt,r) - p(V|g1=1)) = 1 + 1 - p_g1_1 = 2 - p_g1_1
        factor_g1_1 = 2.0 - p_g1_1
        # Factor for g1=0 tokens: (1 + g1(xt,r) - p(V|g1=1)) = 1 + 0 - p_g1_1 = 1 - p_g1_1
        factor_g1_0 = 1.0 - p_g1_1

        # 5. Calculate watermarked probabilities p_wm
        # Initialize p_wm with original probabilities
        p_wm = p.clone()
        # 5. Calculate watermarked probabilities p_wm
        # Initialize p_wm with original probabilities
        p_wm = p.clone()

        # Apply factors based on g-values
        # Ensure mask_g1_1 is boolean
        p_wm[mask_g1_1] *= factor_g1_1
        p_wm[~mask_g1_1] *= factor_g1_0 # Apply factor to g1=0 tokens
        # Apply factors based on g-values
        # Ensure mask_g1_1 is boolean
        p_wm[mask_g1_1] *= factor_g1_1
        p_wm[~mask_g1_1] *= factor_g1_0 # Apply factor to g1=0 tokens

        # 6. Renormalize probabilities to ensure they sum to 1
        # This step is crucial as the formula in Corollary 14 might not
        # perfectly preserve the sum due to floating point inaccuracies
        # or the nature of the transformation.
        p_wm_sum = torch.sum(p_wm)
        if p_wm_sum > epsilon: # Avoid division by zero
            p_wm /= p_wm_sum
        else:
            # Handle case where all probabilities become near zero (should be rare)
            # Fallback to uniform distribution or original probs
            p_wm = torch.ones_like(p_wm) / self.vocab_size
        return p_wm
    
        # 6. Renormalize probabilities to ensure they sum to 1
        # This step is crucial as the formula in Corollary 14 might not
        # perfectly preserve the sum due to floating point inaccuracies
        # or the nature of the transformation.
        p_wm_sum = torch.sum(p_wm)
        if p_wm_sum > epsilon: # Avoid division by zero
            p_wm /= p_wm_sum
        else:
            # Handle case where all probabilities become near zero (should be rare)
            # Fallback to uniform distribution or original probs
            p_wm = torch.ones_like(p_wm) / self.vocab_size
        return p_wm
    
    def get_and_clear_g_values_history(self):
        """Returns the stored g-value history and clears it."""
        history = self.g_values_history
        self.g_values_history = None
        return history


class SynthIDDetector:
    def __init__(self,
                 vocab_size: int,
                 hash_key: int = 15485863,
                 device: torch.device = None,
                 dynamic_seed: str = "markov_1", # "initial", None
                 initial_seed: int = None,
                 expected_g1_proportion: float = 0.5,
                 pval=2e-2):

        self.vocab_size = vocab_size
        self.hash_key = hash_key
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dynamic_seed = dynamic_seed
        self.initial_seed = initial_seed
        self.expected_g1_proportion = expected_g1_proportion
        self.seed_increment = 0 # For fresh randomness if needed
        self.pval = pval
        self.K = 15
        self.K = 15

        if dynamic_seed == "initial" or dynamic_seed is None:
            assert initial_seed is not None, "initial_seed must be provided if dynamic_seed is 'initial' or None."

        # Initialize RNG
        self.rng = torch.Generator(device=self.device)
        if initial_seed is not None and dynamic_seed is None:
             self.rng.manual_seed(self.hash_key * self.initial_seed)


    def _get_seed_for_token(self, prev_token_id: int) -> int:
        """Calculates the seed for the current token based on the dynamic_seed setting."""
        if self.dynamic_seed == "markov_1":
            return self.hash_key * prev_token_id
        elif self.dynamic_seed == "initial":
            return self.hash_key * self.initial_seed
        elif self.dynamic_seed is None:
             # Use the fixed RNG state (seeded once in __init__)
             return 0 # Seed value doesn't matter here
        elif self.dynamic_seed == "fresh":
            # For fresh randomness, we increment the seed based on the current state
            # This is a simple way to ensure each call has a unique seed
            self.seed_increment += 1
            return self.hash_key + self.seed_increment
        else:
            raise ValueError(f"Unknown dynamic_seed type: {self.dynamic_seed}")

    def _get_binary_g_value_for_token(self, token_id: int, seed: int) -> int:
        """
        Generates the binary g-value (0 or 1) for a *specific* token ID based on the seed.
        Must be consistent with the processor's _get_binary_g_values method.
        """
        # Ensure seeding is correct for the chosen dynamic_seed strategy
        if self.dynamic_seed is not None:
             self.rng.manual_seed(seed)

        hashed_values = torch.randint(low = 0, high = self.vocab_size * 10,size = (self.vocab_size,), generator=self.rng, device=self.device)
        g_value = hashed_values[token_id] % 2
        return g_value.item()

    def _compute_z_score(self, observed_g1_count: int, total_tokens: int) -> float:
        """Computes the z-score for the observed count of g1=1 tokens."""
        if total_tokens == 0:
            return 0.0 # Or handle as an error/undefined case

        expected_count = self.expected_g1_proportion * total_tokens
        std_dev = math.sqrt(total_tokens * self.expected_g1_proportion * (1.0 - self.expected_g1_proportion))

        if std_dev == 0:
            # Avoid division by zero if expected proportion is 0 or 1,
            # or if total_tokens is 0 (handled above)
             return float('inf') if observed_g1_count > expected_count else 0.0

        z_score = (observed_g1_count - expected_count) / std_dev
        return z_score

    def detect(self,
               text: str = None,
               tokenized_text: list[int] = None,
               inputs: list[int] = None,
               tokenizer = None,
               return_raw_score: bool = False) -> float:
        
        assert tokenized_text is not None, "Must pass tokenized string"
        
        assert inputs is not None,  "Must pass inputs"

        input_sequence = tokenized_text.tolist()[0]
        prev_token_id = inputs[0][-1].item()


        self.seed_increment=0

        g1_count = 0
        total_tokens = len(input_sequence)
        detected = False
        for i, token_id in enumerate(input_sequence):
            current_seed = self._get_seed_for_token(prev_token_id)
            for _ in range(self.K):
                # SCORE IS ACTUALLY - SUM OF SCORES ALONG ALL LAYER / T*K
                g1_count += self._get_binary_g_value_for_token(token_id, current_seed)
                current_seed += 1
            z = self._compute_z_score(g1_count, self.K*(i+1)) # calculate current zscore
            p = 1-norm.cdf(z)
            if p <= self.pval and not(detected):
                detection_idx = i
                detected = True
            prev_token_id = token_id
        
        if not(detected):
            detection_idx = i

        if return_raw_score:
             return float(g1_count)
        else:
             z_score = self._compute_z_score(g1_count, self.K*total_tokens)
             return z_score, detection_idx

