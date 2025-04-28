import torch
import pdb
from watermark.old_watermark import BlacklistLogitsProcessor
import random
from scipy.stats import ttest_1samp


def top_p_indices(matrix, p):
    """
    Given:
    - a 1D tensor of shape (num_cols,) representing a single probability distribution, OR
    - a 2D tensor of shape (batch_size, num_cols) representing multiple distributions,

    return:
    - if 1D input: a 1D tensor containing the smallest set of indices whose sum is >= p.
    - if 2D input: a list of length batch_size, each element is a 1D tensor of column indices
        for the smallest set of entries in that row whose sum is >= p.

    :param matrix: Probability distribution(s) with shape either (num_cols,) or (batch_size, num_cols).
    :param p: Threshold for cumulative probability (0 < p <= 1).
    :return: A 1D tensor of INDICES (for 1D input) or a list of 1D tensors (for 2D input).
    """
    if matrix.dim() not in (1, 2):
        raise ValueError("Input must be either a 1D or 2D tensor.")

    if matrix.dim() == 1:
        # Single distribution: shape (num_cols,).
        # Convert to shape (1, num_cols) for unified processing,
        # then we'll extract the single row result at the end.
        matrix = matrix.unsqueeze(0)
        single_distribution = True
    else:
        single_distribution = False

    # 1) Sort probabilities (descending) along the last dimension and keep track of original indices
    sorted_probs, sorted_indices = torch.sort(matrix, descending=True, dim=-1)

    # 2) Compute cumulative sums along the sorted probabilities
    cumsum_probs = sorted_probs.cumsum(dim=-1)

    # 3) For each row, find how many entries are needed to reach or exceed p
    #    We do this by checking where cumsum_probs < p, counting those positions,
    #    then taking the next index to ensure >= p.
    keep_lens = (cumsum_probs < p).sum(dim=-1) + 1  # Convert count of "still below p" to index of >= p

    # 4) Gather top-p indices
    results = []
    for row_idx, length in enumerate(keep_lens):
        results.append(sorted_indices[row_idx, :length])

    # For a single distribution, return the single row's result, otherwise return list of results
    if single_distribution:
        return results[0]
    else:
        return results



class InverseTransformLogitsProcessor(BlacklistLogitsProcessor):
    def __call__(self, input_ids, scores):
        """
        Inverse Transform WM logitprocessor.
        Steps:
        0. calc probabilities (softmax of scores)
        1. create seed
        2. set seed
        3. randomly permute vocabulary
        4. create CDF
        5. sample uniform [0,1] rv
        6. find CDF index we exceed the threshold
        7. set that to be the token CDF (set '-inf' to the rest of the scores)
        """
        # pdb.set_trace()
        
        self.permuted_lists = [None for _ in range(input_ids.shape[0])]
        self.u_samples = [None for _ in range(input_ids.shape[0])]

        if self.g_cuda is None:
            self.g_cuda = torch.Generator(device=input_ids.device)

        for b_idx in range(input_ids.shape[0]):
            # seed generation:
            if self.dynamic_seed == "initial":
                seed = self.large_prime*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.large_prime*input_ids[b_idx][-1].item()
            elif self.dynamic_seed is None:
                # let the rng evolve naturally - this is not a realistic setting
                pass
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.large_prime + self.seed_increment
            
            #####
            p = torch.softmax(scores[b_idx], dim=-1)
            self.saved_distributions.append(p.detach().cpu().clone())
            filter_indices = top_p_indices(p, self.top_p)
            p_new = torch.zeros(size=(self.vocab_size,), device=p.device)
            p_new[filter_indices] = p[filter_indices]
            ## normalize
            p_new = p_new / p_new.sum()
            ##
            scores[b_idx] = torch.log(p_new+ 1e-10)  # add small value to avoid log(0)
            #####

            # set seed and sample the random objects:
            self.g_cuda.manual_seed(seed)
            permuted_list = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.g_cuda)
            self.g_cuda.manual_seed(seed)
            u = torch.rand(1, device=input_ids.device, generator=self.g_cuda)

            self.permuted_lists[b_idx] = permuted_list
            self.u_samples[b_idx] = u        

        if not self.noop_blacklist:
            # we choose to watermark
            for b_idx in range(input_ids.shape[0]):
                # 1. create CDF:
                distribution = torch.softmax(scores, dim=-1)
                cdf = torch.cumsum(distribution[b_idx, self.permuted_lists[b_idx]], dim=-1)
                # 2. sample uniform [0,1] rv
                u = self.u_samples[b_idx]
                # print(f'u={u.item()}')
                # 3. find CDF index we exceed the threshold
                idx = torch.searchsorted(cdf, u)
                # 4. set that to be the token CDF (set '-inf' to the rest of the scores)
                scores[b_idx, :] = -float('inf')
                scores[b_idx, self.permuted_lists[b_idx][idx]] = 0

        return scores
    

class InverseTransformDetector():
    def __init__(self,
                 tokenizer,
                 vocab: list[int] = None,
                 gamma: float = 0.5,
                 delta: float = 5.0,
                 hash_key: int = 15485863,
                 initial_seed: int=None,
                 dynamic_seed: str=None, # "initial", "markov_1", None
                 device: torch.device = None,
                 select_green_tokens: bool = True):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hash_key = hash_key
        self.device = device
        self.tokenizer = tokenizer
        self.seed_increment = 0

        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed
        self.rng = torch.Generator(device=self.device)

    def detect(self,
               inputs: list[int]=None,
               tokenized_text: list[int]=None,
               debug: bool=True,
               return_scores: bool = True,):
        
        """
        Implementation of the T-test for detection of Inverse Transform WM.
        """
        # pdb.set_trace()
        assert tokenized_text is not None, "Must pass tokenized string"
        
        assert inputs is not None,  "Must pass inputs"

        input_sequence = tokenized_text.tolist()[0]
        
        # print("input sequence is:", input_sequence)
        prev_token = inputs[0][-1].item()

        detect_scores = []
        self.seed_increment=0
        for idx, tok_gend in enumerate(input_sequence):
            if self.dynamic_seed == "initial":
                seed = self.hash_key*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.hash_key*prev_token
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.hash_key + self.seed_increment


            self.rng.manual_seed(seed)
            permuted_list = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
            self.rng.manual_seed(seed)
            u = torch.rand(1, device=self.device, generator=self.rng)
            # print("u is:", u.item())

            inv = torch.argsort(permuted_list)
            rank = inv[tok_gend].item()
            score = abs(u - rank / (self.vocab_size - 1))
            detect_scores.append(score.item())

            prev_token = tok_gend
        
        # pdb.set_trace()
        # sum_score = sum(detect_scores)
        t_stat, p_value = ttest_1samp(detect_scores, popmean=1/3)
        z_score = [0]
    
        return t_stat, p_value, z_score










        





