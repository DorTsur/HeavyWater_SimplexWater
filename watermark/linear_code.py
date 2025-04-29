import torch
import pdb
import ot
from watermark.old_watermark import BlacklistLogitsProcessor, top_p_indices
import math
import numpy as np

class LinearCodeLogitsProcessor(BlacklistLogitsProcessor):
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
                tilt=False,
                tilting_delta=0.1,
                top_p = 0.999,
                context=1,
                hashing='min'
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
                noop_blacklist)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_params()
        self.tilt = tilt
        self.tilting_delta = tilting_delta
        self.seed_increment = 0
        self.saved_distributions = []
        self.top_p = top_p
        self.context = context
        self.hashing = hashing
        # self.gen_cost(device=device)
    
    def gen_params(self):
        m = self.vocab_size
        self.n = int(torch.ceil(torch.log2(torch.tensor(m, dtype=torch.float32))).item())
        self.k = 2**self.n - 1
        z=2
        if 2 ** self.n != int(m):
            # print("m must be a power of 2")
            # padding = 2 ** n - int(m)
            self.m = 2 ** self.n
        else:
            self.m = m
        
        self.G = self.generate_generator_matrix(device=self.device).to(torch.float)
         
    def gen_seed(self, token_ids):
        pdb.set_trace()
        token_ids = token_ids.tolist()
        if self.hashing == 'min':
            agg = min(token_ids)
        elif self.hashing == 'sum':
            agg = sum(token_ids)
        elif self.hashing == 'prod':
            agg = 1
            for i in token_ids:
                agg *= i
        elif self.hashing == 'repeat':
            token_ids = tuple(token_ids)
            if token_ids in self.hash_dict:
                pass
            else:
                self.hash_dict[token_ids] = self.seed_increment
                self.seed_increment += 1
                agg = 1
                for i in token_ids:
                    agg *= i
        return agg
            
    def __call__(self, input_ids, scores):
        """
        Linear Code WM logitprocessor - currently Simplex.
        Steps:
        1. sample s uniformly over {1, (self.vocab_size)}
        2. gen p = distr via softmax
        3. p_new - tilt_q_sC(p, s, params)
        4. p_new into logits
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
            elif self.dynamic_seed == 'agg_hash':
                #TD - create vrious sliding window hashes.
                seed = self.gen_seed(input_ids[b_idx][-self.context:])

            # set seed and generate s
            self.g_cuda.manual_seed(seed)
            s = torch.randint(
                low=1,
                high=self.m,
                size=(1,),
                generator=self.g_cuda,
                device=input_ids.device
            ).item()  
            # print(f's={s},seed={seed}')     

        if not self.noop_blacklist:
            # we choose to watermark
            scores_new = [None for _ in range(input_ids.shape[0])]
            for b_idx in range(input_ids.shape[0]):
                p = torch.softmax(scores[b_idx], dim=-1)
                # Save a clone so future changes don't affect it
                self.saved_distributions.append(p.detach().cpu().clone())
                p_new =  self.tilt_q_SC(distribution=p, side_info=s)
                scores_new[b_idx] = torch.log(p_new)
            scores_new = torch.stack(scores_new,axis=0)
        return scores_new
    
    def tilt_q_SC(self, distribution, side_info, reg_const=0.05, num_iter=2000, top_k=100, top_p=0.999):
        # reg_const = 0.5
        # num_iter=20
        # top-p filtering:
        # pdb.set_trace()
        # apply top_p filtering to the scores
        filter_indices = top_p_indices(distribution, top_p)
        if len(filter_indices) == 1:
            # then there's nothing to watermark.
            return distribution
        distribution = distribution[filter_indices]
        ## normalize
        distribution = distribution / distribution.sum()
        ##
        # distribution = torch.cat((distribution, torch.zeros(self.padding, device=distribution.device)))
        ps = torch.ones(self.k, device=distribution.device) / self.k  # uniform over S
        # generator matrix and bit matrix
        # G = self.generate_generator_matrix(device=distribution.device).to(torch.float)        # should return torch.Tensor [m x (m-1)]
        matrix = self.int_to_bit_matrix(token_ids=filter_indices,device=distribution.device).to(torch.float)           # should return torch.Tensor [m x log2(m)]
        # compute cost matrix modulo z
        # pdb.set_trace()
        C_orig = matrix @ self.G % 2
        C = 1-C_orig 
        # reduce cost matrix
        # reduced = reduce_cost_matrix(distribution, ps, C, top_k=top_k, top_p=top_p)
        # reduced = {
        #     'C_reduced': self.C,
        #     'ps_merged': ps,
        #     'px_reduced': distribution
        # }
        # sinkhorn on reduced cost
        P = ot.sinkhorn(
            a=distribution,
            b=ps,
            M=C,
            reg=reg_const,
            numItermax=num_iter,
            stopThr=1e-5
        )
        P_wm = P[:, side_info - 1]
        if self.tilt:
            # pdb.set_trace()
            c = C_orig[ :,side_info - 1]
            index_1s = torch.where(c == 1)[0]
            index_0s = torch.where(c == 0)[0]
            P_wm[index_1s] = P_wm[index_1s]*(1+self.tilting_delta)
            P_wm[index_0s] = P_wm[index_0s]*(1-self.tilting_delta)
        # reconstruct conditional
        # P_wm = torch.zeros(m, device=distribution.device)
        # merged_col = reduced['inverse'][side_info - 1]
        # P_wm[reduced['keep_indices']] = P_reduced[:, merged_col]
        P_wm = P_wm / P_wm.sum()
        out_p = torch.zeros(size=(self.vocab_size,), device=distribution.device)
        out_p[filter_indices] = P_wm

        
        
        return out_p
    
    def generate_generator_matrix(self, device=None):
        # pdb.set_trace()
        # Generate all binary vectors of length n (excluding 0)
        all_columns = torch.stack([
            torch.tensor([int(bit) for bit in format(i, f'0{self.n}b')], device=device)
            for i in range(1, self.m)
        ])
        return all_columns.T  # Shape: (n, m-1)

    def int_to_bit_matrix(self, token_ids, device=None):
        
        return torch.stack([
            torch.tensor([int(bit) for bit in format(i, f'0{self.n}b')], device=device)
            for i in token_ids
        ])
    

class LinearCodeWatermarkDetector():
    """
    Class for detecting watermarks
    """
    
    def __init__(self,
                 tokenizer,
                 vocab: list[int] = None,
                 gamma: float = 0.5,
                 delta: float = 5.0,
                 hash_key: int = 15485863,
                 initial_seed: int=None,
                 dynamic_seed: str=None, # "initial", "markov_1", None
                 device: torch.device = None,
                 select_green_tokens: bool = True,
                 
                 ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
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

        m = self.vocab_size
        self.n = int(torch.ceil(torch.log2(torch.tensor(m, dtype=torch.float32))).item())
        z=2
        if 2 ** self.n != int(m):
            # print("m must be a power of 2")
            # padding = 2 ** n - int(m)
            self.m = 2 ** self.n
        
        # self.gen_rand_G(device=self.device)

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count =  0.5
        print(f"detected {observed_count} tokens out of {T}")
        numer = observed_count - expected_count * T
        denom = math.sqrt(T * expected_count * (1 - expected_count))
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
        cnt=0
        self.seed_increment = 0
        for idx, tok_gend in enumerate(input_sequence):
            if self.dynamic_seed == "initial":
                seed = self.hash_key*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.hash_key*prev_token
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.hash_key + self.seed_increment
            
            self.rng.manual_seed(seed)
            s = torch.randint(
                low=1,
                high=self.m,
                size=(1,),
                generator=self.rng,
                device=self.device
            ).item()

            # print(f's={s},token={tok_gend}, prev_token={prev_token}')

            binary_x = torch.tensor([int(bit) for bit in format(tok_gend, f'0{self.n}b')], device=self.device)
            binary_s = torch.tensor([int(bit) for bit in format(s, f'0{self.n}b')], device=self.device)
            
            # binary generator:
            cnt += (binary_x.to(torch.float32) @ binary_s.to(torch.float32) % 2).item()
            
            
            prev_token = tok_gend
            
        # pdb.set_trace()
        z_score = self._compute_z_score(cnt, len(input_sequence))
        print("LC z score is:", z_score)
        return z_score

