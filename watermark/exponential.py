import torch
import torch.nn.functional as F
import pdb
from watermark.old_watermark import BlacklistLogitsProcessor
import math
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
                noop_blacklist= False
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
                noop_blacklist
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

        # set seed and generate random number
        self.g_cuda.manual_seed(seed)
        random_values = torch.rand(size=(self.vocab_size,), \
            generator=self.g_cuda, device=self.device)
        print(f's={random_values},seed={seed}')     

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
        for idx, tok_gend in enumerate(input_sequence):
            if self.dynamic_seed == "initial":
                seed = self.hash_key*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.hash_key*prev_token
            
            self.rng.manual_seed(seed)
            random_values = torch.rand(size=(self.vocab_size,), \
                generator=self.rng, device=self.device)

            #print(f'random values={random_values},token={tok_gend}, prev_token={prev_token}')

            # compute test statistic
            teststats += -torch.log(torch.clamp(1 - random_values[tok_gend], min=1e-5))
            prev_token = tok_gend
            
        # pdb.set_trace()
        z_score = self._compute_z_score(teststats, len(input_sequence))
        print(f"z_score={z_score}, teststats={teststats}")
        print(f"teststats bias: {teststats - len(input_sequence)}")
        return z_score.item(), teststats.item()