import torch
import torch.nn.functional as F
import pdb
from watermark.old_watermark import BlacklistLogitsProcessor
import math

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



class ExponentialLogitsProcessor(BlacklistLogitsProcessor):
    def __init__(self, 
                bad_words_ids, 
                eos_token_id,
                vocab, 
                vocab_size,
                bl_proportion=0.5,
                bl_logit_bias=1.0,
                bl_type= "hard", # "soft"
                initial_seed=None, 
                dynamic_seed=None, # "initial", "markov_1", None
                store_bl_ids=False,
                store_spike_ents= False,
                noop_blacklist= False,
                top_p=0.9,
                ):
        super().__init__(bad_words_ids, 
                eos_token_id,
                vocab, 
                vocab_size,
                bl_proportion,
                bl_logit_bias,
                bl_type, # "soft"
                initial_seed, 
                dynamic_seed, # "initial", "markov_1", None
                store_bl_ids,
                store_spike_ents,
                noop_blacklist,
                top_p=top_p,
                )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def __call__(self, input_ids, scores):
        """
        Exponential WM logitprocessor 
        Steps:
        1. get seeds from random number generator 
        2. compute hash 
        3. sample token by Gumbel trick (exponential sampling)
        4. update logits
        """
        # pdb.set_trace()

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

        # set seed and generate random number
        self.g_cuda.manual_seed(seed)
        random_values = torch.rand(size=(self.vocab_size,), \
            generator=self.g_cuda, device=self.device)
        # print(f's={random_values},seed={seed}')     

        # Compute probabilities and get random values
        probs = F.softmax(scores, dim=-1)
        hash_values = torch.div(-torch.log(random_values), probs)

        # Get next token, and update logit
        next_token = hash_values.argmin(dim=-1)

        # Update scores
        scores[:] = -math.inf
        scores[torch.arange(scores.shape[0]), next_token] = 0
        return scores
    

class ExponentialWatermarkDetector():
    """
    Class for detecting watermarks
    """
    
    def __init__(self,
                 tokenizer,
                 vocab: list[int] = None,
                #  gamma: float = 0.5,
                #  delta: float = 5.0,
                 hash_key: int = 15485863,
                 initial_seed: int=None,
                 dynamic_seed: str=None, # "initial", "markov_1", None
                 device: torch.device = None,
                 select_green_tokens: bool = True,
                 
                 ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # self.gamma = gamma
        # self.delta = delta
        self.hash_key = hash_key
        self.device = device
        self.tokenizer = tokenizer
        self.select_green_tokens = select_green_tokens
        self.seed_increment = 0
        
        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed
        self.rng = torch.Generator(device=self.device)


    def _compute_z_score(self, observed_count, T):
        mean = T
        var = T
        numer = observed_count - T
        denom = math.sqrt(T)
        z = numer / denom
        return z
    
    def detect(self,
               inputs: list[int]=None,
               tokenized_text: list[int]=None,
               debug: bool=True,
               return_scores: bool = True,):
        assert tokenized_text is not None, "Must pass tokenized string"
        assert inputs is not None,  "Must pass inputs"


        input_sequence = tokenized_text.tolist()[0]
        prev_token = inputs[0][-1].item()
        teststats=0
        gen_token_length = 0
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
            random_values = torch.rand(size=(self.vocab_size,), \
                generator=self.rng, device=self.device)

            #print(f'random values={random_values},token={tok_gend}, prev_token={prev_token}')

            # compute test statistic
            teststats += -torch.log(torch.clamp(1 - random_values[tok_gend], min=1e-5))
            prev_token = tok_gend
            gen_token_length += 1

            
        # pdb.set_trace()
        # teststats ~ Gamma(T,1), mean T and Var T. Hence, we can approximate it as normal with mean T and variance T.
        z_score = self._compute_z_score(teststats, len(input_sequence)) 
        print(f"z_score={z_score}, teststats={teststats}")
        print(f"teststats bias: {teststats - len(input_sequence)}")
        return z_score.item(), teststats.item(),gen_token_length