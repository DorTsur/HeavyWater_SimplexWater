# Implementation of SynthID-Text watermarking (Vectorized Binary N=2)
# Based on Corollary 14 in https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf

from typing import List, Optional

import math
import torch
import random
import numpy as np

from torch import Tensor
from transformers import LogitsProcessor

class SynthIDLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor for applying the SynthID watermark using the
    vectorized binary approach (N=2) described in Corollary 14.

    Args:
        vocab_size (`int`):
            The size of the vocabulary.
        hash_key (`int`):
            Key used for seeding the hash function.
        device (`torch.device`):
            The device to run the calculations on (e.g., 'cuda' or 'cpu').
        dynamic_seed (`str`, *optional*, defaults to `"markov_1"`):
            How to seed the random number generator.
            "markov_1": Seed based on the previous token.
            "initial": Seed based on a fixed initial seed (requires `initial_seed`).
            None: Use a fixed seed (requires `initial_seed`). Not recommended for security.
        initial_seed (`int`, *optional*, defaults to `None`):
            Initial seed value, required if `dynamic_seed` is "initial" or None.
        store_g_values (`bool`, *optional*, defaults to `False`):
            Whether to store the generated g-values for debugging/analysis.
    """

    def __init__(self,
                 vocab_size: int,
                 hash_key: int = 15485863, # Default from Kirchenbauer et al.
                 device: torch.device = None,
                 dynamic_seed: str = "markov_1", # "initial", None
                 initial_seed: int = None,
                 store_g_values: bool = False
                 ):

        self.vocab_size = vocab_size
        self.hash_key = hash_key
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dynamic_seed = dynamic_seed
        self.initial_seed = initial_seed
        self.store_g_values = store_g_values
        self.g_values_history = None # To store g-values if store_g_values is True

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
        batch_size = scores.shape[0]
        new_scores = torch.zeros_like(scores)

        if self.store_g_values:
            if self.g_values_history is None:
                self.g_values_history = [[] for _ in range(batch_size)]

        # Calculate current probabilities from logits
        probs = torch.softmax(scores, dim=-1)

        for b_idx in range(batch_size):
            # 1. Get the seed for the current step
            current_seed = self._get_seed_for_token(input_ids, b_idx)

            # 2. Generate binary g-values (0 or 1) for the vocabulary
            g1_values = self._get_binary_g_values(current_seed) # Shape [vocab_size]

            if self.store_g_values:
                 # Store g-values for this step and batch index (optional)
                 # Store the full g-value vector or just for the sampled token later
                 self.g_values_history[b_idx].append(g1_values.clone().cpu())


            # 3. Calculate p(V | g1=1) = sum of probabilities of tokens with g1=1
            # Ensure g1_values is boolean or compatible type for indexing/masking
            mask_g1_1 = (g1_values == 1)
            p_g1_1 = torch.sum(probs[b_idx][mask_g1_1])

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
            p_wm = probs[b_idx].clone()

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


            # 7. Convert watermarked probabilities back to logits
            # Add epsilon to avoid log(0)
            new_scores[b_idx] = torch.log(p_wm + epsilon)

        return new_scores

    def get_and_clear_g_values_history(self):
        """Returns the stored g-value history and clears it."""
        history = self.g_values_history
        self.g_values_history = None
        return history


class SynthIDDetector:
    """
    Detector for the SynthID watermark (Vectorized Binary N=2).

    Args:
        vocab_size (`int`):
            The size of the vocabulary.
        hash_key (`int`):
            Key used for seeding the hash function (must match the processor).
        device (`torch.device`):
            The device to run the calculations on (e.g., 'cuda' or 'cpu').
        dynamic_seed (`str`, *optional*, defaults to `"markov_1"`):
            How to seed the random number generator (must match the processor).
            "markov_1": Seed based on the previous token.
            "initial": Seed based on a fixed initial seed (requires `initial_seed`).
             None: Use a fixed seed (requires `initial_seed`). Not recommended for security.
        initial_seed (`int`, *optional*, defaults to `None`):
            Initial seed value, required if `dynamic_seed` is "initial" or None.
        expected_g1_proportion (`float`, *optional*, defaults to `0.5`):
            The expected proportion of tokens with g-value 1 under the null hypothesis
            (unwatermarked text). Should match the hashing scheme used.
    """
    def __init__(self,
                 vocab_size: int,
                 hash_key: int = 15485863,
                 device: torch.device = None,
                 dynamic_seed: str = "markov_1", # "initial", None
                 initial_seed: int = None,
                 expected_g1_proportion: float = 0.5):

        self.vocab_size = vocab_size
        self.hash_key = hash_key
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dynamic_seed = dynamic_seed
        self.initial_seed = initial_seed
        self.expected_g1_proportion = expected_g1_proportion

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

        # Replicate the hashing logic from the processor for a single token
        # We need to generate the value for the specific token_id
        # A simple way is to generate for all and pick, but that's inefficient.
        # Let's refine the hashing to be per-token if possible.
        # If using torch.randint as in processor, we might need to generate
        # up to token_id+1 values, which is still inefficient.
        # A direct hash function is better for detection:
        # simple_hash = (token_id * self.hash_key + seed) % (self.vocab_size * 10)
        # g_value = simple_hash % 2
        # For consistency with the processor's current RNG method:
        hashed_values = torch.randint(0, self.vocab_size * 10, (token_id + 1,),
                                      generator=self.rng, device=self.device)
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
               prompt_tokens: list[int] = None,
               tokenizer = None,
               return_raw_score: bool = False) -> float:
        """
        Detects the SynthID watermark in the given text.

        Args:
            text (`str`, *optional*): The text sequence to analyze.
            tokenized_text (`list[int]`, *optional*): Pre-tokenized text sequence.
            prompt_tokens (`list[int]`, *optional*): The tokenized prompt used to generate the text.
                                                     Needed for 'markov_1' seeding.
            tokenizer: The tokenizer used for the model. Required if `text` is provided.
            return_raw_score (`bool`, *optional*, defaults to `False`):
                 If True, returns the raw count of g1=1 tokens instead of the z-score.


        Returns:
            `float`: The z-score indicating the likelihood of the text being watermarked.
                     Higher scores suggest a higher likelihood. If return_raw_score is True,
                     returns the integer count of g1=1 tokens.
        """
        assert (text is not None and tokenizer is not None) or (tokenized_text is not None), \
               "Either text and tokenizer or tokenized_text must be provided."
        assert self.dynamic_seed != "markov_1" or prompt_tokens is not None, \
               "prompt_tokens must be provided for dynamic_seed='markov_1'."

        if tokenized_text is None:
            tokenized_text = tokenizer.encode(text, add_special_tokens=False)

        if not tokenized_text:
             return 0.0 # No tokens to analyze

        # Determine the starting token for seeding
        if self.dynamic_seed == "markov_1":
            if not prompt_tokens:
                 # Maybe raise error or use a default if prompt is empty/not given
                 # For now, assume a default start token ID like 0 if prompt is empty
                 prev_token_id = 0
            else:
                 prev_token_id = prompt_tokens[-1]
        else:
             # For 'initial' or None seed, the first token's seed is fixed
             prev_token_id = -1 # Placeholder, not used directly for seeding in these cases


        g1_count = 0
        total_tokens = len(tokenized_text)

        for i, token_id in enumerate(tokenized_text):
            # Get seed based on the *previous* token
            current_seed = self._get_seed_for_token(prev_token_id)

            # Get the g-value for the *current* token using that seed
            g1_value = self._get_binary_g_value_for_token(token_id, current_seed)

            if g1_value == 1:
                g1_count += 1

            # Update previous token for the next iteration
            prev_token_id = token_id

        if return_raw_score:
             return float(g1_count)
        else:
             z_score = self._compute_z_score(g1_count, total_tokens)
             return z_score


# Example Usage (Conceptual - requires tokenizer, model context)
if __name__ == '__main__':
    # Dummy values for demonstration
    vocab_size = 50257 # Example: GPT-2 vocab size
    hash_key = 12345
    initial_seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Processor Example ---
    print("--- Logits Processor Example ---")
    processor = SynthIDLogitsProcessor(
        vocab_size=vocab_size,
        hash_key=hash_key,
        device=device,
        dynamic_seed="markov_1", # Use previous token for seed
        initial_seed=initial_seed # Needed for the very first token if input_ids len is 1
    )

    # Dummy input logits and input_ids
    batch_size = 2
    seq_len = 10
    dummy_scores = torch.randn(batch_size, vocab_size, device=device)
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print(f"Original logits shape: {dummy_scores.shape}")
    print(f"Input IDs shape: {dummy_input_ids.shape}")

    # Apply the watermark
    watermarked_scores = processor(dummy_input_ids, dummy_scores)
    print(f"Watermarked logits shape: {watermarked_scores.shape}")

    # Verify probabilities sum roughly to 1 (after softmax)
    watermarked_probs = torch.softmax(watermarked_scores, dim=-1)
    print(f"Sum of watermarked probabilities (batch 0): {watermarked_probs[0].sum()}")
    print(f"Sum of watermarked probabilities (batch 1): {watermarked_probs[1].sum()}")
    print("-" * 20)


    # --- Detector Example ---
    print("\n--- Detector Example ---")
    detector = SynthIDDetector(
        vocab_size=vocab_size,
        hash_key=hash_key, # Must match processor
        device=device,
        dynamic_seed="markov_1", # Must match processor
        initial_seed=initial_seed # Must match processor if needed
    )

    # Dummy generated text (token IDs) and prompt
    dummy_prompt_tokens = [101, 500, 600] # Example token IDs
    dummy_generated_tokens = [1000, 2000, 3000, 1500, 2500, 3500, 4000, 1200, 2200, 3200] * 2 # Longer sequence

    print(f"Prompt tokens: {dummy_prompt_tokens}")
    print(f"Generated tokens length: {len(dummy_generated_tokens)}")

    # Detect the watermark
    z_score = detector.detect(
        tokenized_text=dummy_generated_tokens,
        prompt_tokens=dummy_prompt_tokens
    )
    print(f"Calculated z-score: {z_score:.4f}")

    # Example with raw score
    raw_score = detector.detect(
        tokenized_text=dummy_generated_tokens,
        prompt_tokens=dummy_prompt_tokens,
        return_raw_score=True
    )
    print(f"Raw g1=1 count: {raw_score}")
    print("-" * 20)

    # Example with initial seed dynamic seeding
    print("\n--- Detector Example (Initial Seed) ---")
    detector_initial = SynthIDDetector(
        vocab_size=vocab_size,
        hash_key=hash_key,
        device=device,
        dynamic_seed="initial", # Use fixed initial seed
        initial_seed=initial_seed # Must provide seed
    )
    z_score_initial = detector_initial.detect(
        tokenized_text=dummy_generated_tokens,
        prompt_tokens=dummy_prompt_tokens # Prompt still needed for context, but not for seeding after first token
    )
    print(f"Calculated z-score (initial seed): {z_score_initial:.4f}")