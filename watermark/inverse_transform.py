import torch
import pdb
from watermark.old_watermark import BlacklistLogitsProcessor
import random
from scipy.stats import ttest_1samp



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
        pdb.set_trace()
        
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
            
            
            # set seed and sample the random objects:
            self.g_cuda.manual_seed(seed)
            permuted_list = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.g_cuda)
            self.g_cuda.manual_seed(seed)
            u = torch.rand(1)

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
        
        assert tokenized_text is not None, "Must pass tokenized string"
        
        assert inputs is not None,  "Must pass inputs"

        input_sequence = tokenized_text.tolist()[0]
        
        # print("input sequence is:", input_sequence)
        prev_token = inputs[0][-1].item()

        detect_scores = []
        for idx, tok_gend in enumerate(input_sequence):
            if self.dynamic_seed == "initial":
                seed = self.hash_key*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.hash_key*prev_token


            self.rng.manual_seed(seed)
            permuted_list = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
            self.rng.manual_seed(seed)
            u = torch.rand(1, device=self.device)
            print("u is:", u)

            inv = torch.argsort(permuted_list)
            rank = inv[tok_gend].item()
            score = abs(u - rank / (self.vocab_size - 1))
            detect_scores.append(score)

            prev_token = tok_gend
        
        sum_score = sum(detect_scores)
        t_stat, p_value = ttest_1samp(detect_scores, popmean=1/3)










        





