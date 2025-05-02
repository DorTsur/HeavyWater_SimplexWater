import torch
import pdb
import ot
from watermark.old_watermark import BlacklistLogitsProcessor, top_p_indices
import math
import numpy as np
import itertools
from scipy.stats import power_divergence, chisquare
from numpy.random import default_rng
from tqdm import tqdm

# Compute log with base q
def log_base_q(x, q):
    return np.log(x) / np.log(q)

def int_to_base_q_vec(i: int, n: int, q: int, device=None):
    """
    Return a length-n LongTensor whose entries are the base-q digits of i.
    Most-significant digit first.  No Python loops on the hot path.
    """
    i = torch.as_tensor(i, device=device, dtype=torch.long)
    powers = q ** torch.arange(n - 1, -1, -1, device=device)   # q^{n-1},…,q^0
    return ((i.unsqueeze(-1) // powers) % q)

class Q_LinearCodeLogitsProcessor(BlacklistLogitsProcessor):
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
                hashing='min',
                q=2
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
        # pdb.set_trace()
        self.q = q
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
        # pdb.set_trace()
        m = self.vocab_size
        self.n = int(np.ceil(log_base_q(m,self.q)).item())
        if self.q ** self.n != int(m):
            self.m = self.q ** self.n
        else:
            self.m = m
        self.k = self.m-1
        self.G = self.qary_hadamard_generator(self.q, self.m, include_constant=False, device=self.device)

            
    def gen_seed(self, token_ids):
        # pdb.set_trace()
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
        # pdb.set_trace()
        filter_indices = top_p_indices(distribution, top_p)
        # FILTER OUT ID=0
        if len(filter_indices) == 1:
            # then there's nothing to watermark.
            return distribution
        distribution = distribution[filter_indices]
        distribution = distribution / distribution.sum()
        ps = torch.ones(self.k, device=distribution.device) / self.k  # uniform over S
        matrix = self.int_to_log_q_matrix(token_ids=filter_indices,device=distribution.device)
        # C_orig = ((matrix.to(torch.float32)) @ (self.G.to(torch.float32)) % self.q)
        # C_orig = ((matrix.to(torch.float16)) @ (self.G.to(torch.float16)) % self.q)/float(self.q-1)
        C_orig = ((matrix.to(torch.float32)) @ (self.G.to(torch.float32)) % self.q)/float(self.q-1)
        C = 1-C_orig 
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
            # print(f'tilting')
            # pdb.set_trace()
            thresh = 0.5
            c = C_orig[ :,side_info - 1]
            ##
            # index_up = torch.where(c >= thresh)[0]
            # index_down = torch.where(c < thresh)[0]
            # P_wm[index_up] = P_wm[index_up]*(1+self.tilting_delta)
            # P_wm[index_down] = P_wm[index_down]*(1-self.tilting_delta)
            ##
            P_wm = P_wm*torch.exp(self.tilting_delta*(c-thresh))
        P_wm = P_wm / P_wm.sum()
        out_p = torch.zeros(size=(self.vocab_size,), device=distribution.device)
        # out_p = torch.zeros(size=(self.vocab_size,), device=distribution.device, dtype=P_wm.dtype)
        out_p[filter_indices] = P_wm
        
        return out_p
        
        
    
    def qary_hadamard_generator(self, q, m, include_constant=False,device=None):
        # points = list(itertools.product(range(q), repeat=m))
        # rows = [[p[i] for p in points] for i in range(m)]
        # if include_constant:
        #     rows.insert(0, [1]*len(points))
        # G = np.array(rows) % q
        # return torch.tensor(G, dtype=torch.float32, device=device)
        # pdb.set_trace()
        return torch.stack([int_to_base_q_vec(i, self.n, self.q, device=device) for i in range(1, m)]).T
                # torch.tensor([int(bit) for bit in format(i, f'0{self.n}b')], device=device)
                # for i in token_ids
            

    
    def int_to_log_q_matrix(self, token_ids, device=None):
        token_ids_ = [i-1 for i in token_ids if i != 0]
        if 0 in token_ids:
            # pdb.set_trace()
            matrix = self.G[:,token_ids_]
            zero_vec = int_to_base_q_vec(0, self.n, self.q, device=device).unsqueeze(0).T
            matrix = torch.concatenate([zero_vec,matrix], axis=1).T
        else:
            matrix =  self.G[:,token_ids_].T
        # matrix =  torch.stack([int_to_base_q_vec(i, self.n, self.q, device=device) for i in token_ids])
        
        # if torch.any(matrix != matrix_):
        #     print("discrepancy in matrices")
        return matrix

        
    
    def generate_generator_matrix(self, device=None):
        #####
        # # pdb.set_trace()
        # Generate all binary vectors of length n (excluding 0)
        all_columns = torch.stack([ torch.tensor([int(bit) for bit in format(i, f'0{self.n}b')], device=device) for i in range(1, self.m)])
        return all_columns.T  # Shape: (n, m-1)

    def int_to_bit_matrix(self, token_ids, device=None):
        return torch.stack([torch.tensor([int(bit) for bit in format(i, f'0{self.n}b')], device=device) for i in token_ids])
    
    
    
    def int_to_base_q(self, i, device=None):
        digits = []
        for _ in range(self.n):
            digits.append(i % self.q)
            i //= self.q
        digits.reverse()
        return torch.tensor(digits, device=device)
    

# ---------- 2.  Pearson χ² test (for comparison) ----------
def pearson_chi2(counts):
    counts = np.asarray(counts, dtype=float)
    n, m = counts.sum(), len(counts)
    stat, p = chisquare(counts, f_exp=np.full(m, n/m))
    return stat, p

# ---------- 3.  Exact p-value via Monte Carlo ----------
def g_stat(counts):
    counts = counts[counts > 0]
    return 2 * np.sum(counts * np.log(counts / (counts.sum() / len(counts))))

def multinomial_uniform_exact(counts, n_sim=200000, seed=None):
    """
    Estimates the exact p-value by Monte-Carlo.
    n_sim : number of simulated multinomial tables
    """
    counts = np.asarray(counts, dtype=int)
    n, m = counts.sum(), len(counts)
    obs = g_stat(counts)
    rng = default_rng(seed)
    more_extreme = 0
    for _ in tqdm(range(n_sim)):
        sim = rng.multinomial(n, np.repeat(1/m, m))
        if g_stat(sim) >= obs:
            more_extreme += 1
    # add-one smoothing so p>0 even when more_extreme==0
    p_mc = (more_extreme + 1) / (n_sim + 1)
    return obs, p_mc

class Q_LinearCodeWatermarkDetector():
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
                 q = 2,
                 context=1,
                 hashing='min'
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
        self.q = q
        self.context = context 
        self.hashing = hashing
        
        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed
        self.rng = torch.Generator(device=self.device)

        m = self.vocab_size
        self.n = int(np.ceil(log_base_q(m,self.q)).item())
        if self.q ** self.n != int(m):
            self.m = self.q ** self.n
        else:
            self.m = m
        self.k = self.m-1
        # self.n = int(torch.ceil(torch.log2(torch.tensor(m, dtype=torch.float32))).item())
        # z=2
        # if 2 ** self.n != int(m):
        #     # print("m must be a power of 2")
        #     # padding = 2 ** n - int(m)
        #     self.m = 2 ** self.n
        
        # self.gen_rand_G(device=self.device)

    def g_test(self, counts):
        """
        counts : 1-D integer array of length q (field size), summing to T (generated tokens)
        Returns (G2 statistic, p_value) using χ²_{q-1} null.
        """
        counts = np.asarray(counts, dtype=float)
        T, q = counts.sum(), len(counts)
        expected = T / q
        # power_divergence with lambda_=0 gives the log-likelihood ratio (G-test)
        G2, p = power_divergence(counts, f_exp=np.full(q, expected), lambda_="log-likelihood")
        #G2, p = power_divergence(counts, f_exp=np.full(q, expected), lambda_="pearson")
        #G2, p = multinomial_uniform_exact(counts, n_sim=2000, seed=self.hash_key)
        # G2, p = pearson_chi2(counts)
        return G2, p

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count =  (self.q -1)/2
        var = (self.q**2 - 1)/12
        # print(f"mean: {expected_count}, variance: {var}")
        #print(f"detected {observed_count} tokens out of {T}")
        numer = observed_count - expected_count * T
        denom = math.sqrt(T * var)
        z = numer / denom
        return z
    
    def detect(self,
               inputs: list[int]=None,
               tokenized_text: list[int]=None,
               debug: bool=True,
               return_scores: bool = True,):
        assert tokenized_text is not None, "Must pass tokenized string"
        
        assert inputs is not None,  "Must pass inputs"
        # pdb.set_trace()

        input_sequence = tokenized_text.tolist()[0]
        prev_token = inputs[0][-1].item()
        cnt= [0 for _ in range(self.q)]
        cnt_zscore = 0
        # f_sum_indep = np.zeros(len(input_sequence))
        self.seed_increment = 0
        for idx, tok_gend in enumerate(input_sequence):
            if self.dynamic_seed == "initial":
                seed = self.hash_key*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.hash_key*prev_token
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.hash_key + self.seed_increment
            elif self.dynamic_seed == 'agg_hash':
                # pdb.set_trace()
                #TD - create vrious sliding window hashes.
                seed = self.gen_seed(inputs[0][-self.context:])
                inputs = torch.concatenate([inputs, tok_gend*torch.ones(size=(1,1), dtype=inputs.dtype)], axis=-1)
            
            self.rng.manual_seed(seed)
            s = torch.randint(
                low=1,
                high=self.m,
                size=(1,),
                generator=self.rng,
                device=self.device
            ).item()

            # print(f's={s},token={tok_gend}, prev_token={prev_token}')
            qarray_x = int_to_base_q_vec(tok_gend, self.n, self.q, device=self.device)
            qarray_s = int_to_base_q_vec(s, self.n, self.q, device=self.device)
            # qarray generator: f(x,s)
            temp_f = int((qarray_x.float() @ qarray_s.float() % self.q))
            cnt[temp_f] += 1
            cnt_zscore += temp_f
            # f_sum_indep[idx] = temp_f 
            prev_token = tok_gend
            
        # pdb.set_trace()
        print(cnt)
        chi_square_statistic, p_value = self.g_test(cnt)
        #print(f"Chi-square statistic: {chi_square_statistic}, p-value: {p_value}")
        
        z_score = self._compute_z_score(cnt_zscore, len(input_sequence))
        # f_sum_mean = (self.q - 1) / 2
        # f_sum_st = math.sqrt((self.q**2 - 1) / 12)
        # v = (f_sum_indep - f_sum_mean)/ f_sum_st
        # print('centered z-score stats mean, variance:', v.mean(), v.std())
        #print("Qarray z score is:", z_score)
        # return z_score
        return chi_square_statistic, p_value, z_score
    
    def gen_seed(self, token_ids):
        # pdb.set_trace()
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