# micro lib to implement the watermarking extensions to LM generation
# as well as utils for eval/validaiton

from typing import List, Optional, Callable

import time
import random
import math
import torch
import numpy as np

from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor, LogitsProcessorList, set_seed
import pdb

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



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def tokenize_and_truncate(example: dict,
                          completion_length: int = None,
                          prompt_length: int = None,
                          hf_model_name: str = None,
                          tokenizer = None,
                          model_max_seq_len: int = 4096):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert "text" in example, "expects 'text' field to be present"
    # tokenize
    inputs = tokenizer.encode(example["text"], return_tensors="pt", truncation=True, max_length=model_max_seq_len)
    example.update({"untruncated_inputs": inputs})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs.shape[1]-1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs.shape[1]-1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError((f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                          f" but got completion_length:{completion_length},prompt_length:{prompt_length}"))

    # truncate
    inputs = inputs[:,:inputs.shape[1]-slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name: 
        inputs[0,-1] = 1
    # else: pass
    example.update({"inputs": inputs})
    return example


class CorrelatedChannelLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the token ids of the words
            that should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, 
                bad_words_ids: List[List[int]], 
                eos_token_id: int,
                vocab: list[int], 
                vocab_size: int,
                bl_proportion: float=0.5,
                bl_logit_bias: float=1.0,
                bl_type: str = "hard", # "soft"
                initial_seed: int=None, 
                dynamic_seed: str=None, # "initial", "markov_1", None
                store_bl_ids: bool=False,
                store_spike_ents: bool = False,
                noop_blacklist: bool = False,
                cc_k=2,
                tilt: bool = False,
                tilting_delta: float = 0.1,
                top_p: float = 0.9,
                ):
        self.top_p = top_p
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.bl_proportion = bl_proportion
        self.bl_logit_bias = bl_logit_bias
        self.bl_type = bl_type
        self.k = cc_k
        self.tilt = tilt
        self.tilting_delta = tilting_delta
        self.seed_increment = 0

        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed

        self.eos_token_id = eos_token_id
        # self.bad_words_id_length_1 = self._prepare_bad_words(bad_words_ids)

        self.bad_words_mask: Optional[torch.LongTensor] = None

        self.store_bl_ids = store_bl_ids
        self.bl_ids = None

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None

        # hack to replace this with an approximation of infinite bias
        # so that the expectation coefficient will come out to 1.0
        if self.bl_type == "hard":
            self.bl_logit_bias = 10000 # FIXME to a value that is actually close to the largest soft watermark used

        alpha = torch.exp(torch.tensor(self.bl_logit_bias)).item()
        # gamma = self.bl_proportion
        gamma = 1.0-self.bl_proportion
        self.alpha = alpha
        self.gamma = gamma

        self.z_value = ((1-gamma)*(alpha-1))/(1-gamma+(alpha*gamma))
        self.expected_wl_coef = (gamma*alpha)/(1-gamma+(alpha*gamma))

        # catch for overflow when bias is "infinite"
        if self.alpha == torch.inf:
            self.z_value = 1.0
            self.expected_wl_coef = 1.0

        self.noop_blacklist = noop_blacklist
        if self.noop_blacklist: print(f"Blacklist processor for accounting only, no rescoring of logits")

        self.g_cuda = None
        self.large_prime = 15485863

        self.side_info = []

    @property
    def blacklisted_ids(self):
        assert self.store_bl_ids, "Need to instantiate processor with `store_bl_ids` to be able to retrieve them later"
        # flatten the each indexes blacklist
        l_of_bl_ids = [[] for _ in range(len(self.bl_ids))]
        for b_idx, batch in enumerate(self.bl_ids):
            for l_of_l, seed in batch:
                bl_ids = [l[0] for l in l_of_l] # this was the main line, maybe unnecessary now?
                l_of_bl_ids[b_idx].append((bl_ids,seed))
        return l_of_bl_ids

    def get_and_clear_stored_bl_ids(self):
        old_bl_ids = self.bl_ids
        self.bl_ids = None
        return old_bl_ids

    def get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def get_and_clear_stored_spike_ents(self):
        spike_ents = self.get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1+(self.z_value*probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.k = 2
        #initialize bl_ids:
        self.bad_words_id_length_1 = [None for _ in range(input_ids.shape[0])]

        #initialize generator if needed
        # pdb.set_trace()
        if self.g_cuda is None:
            self.g_cuda = torch.Generator(device=input_ids.device)
            print(f"input id device {input_ids.device}")
            # self.g_cuda = torch.Generator(device='cuda')

        side_info = []
        # loop over elements in the batch:
        for b_idx in range(input_ids.shape[0]):
            # pdb.set_trace()
            # set seed:
            if self.dynamic_seed == "initial":
                seed = self.large_prime*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.large_prime*input_ids[b_idx][-1].item()
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.large_prime + self.seed_increment
            # print(f'seed={seed}')
            
            
            #create the list (Bm) - CURRENTLY BINARY
            ##
            # The way the list is crearted is by createing a random permutation of all token ids and taking the first half of it
            # needs to be rethought for k>2. 
            ##
            # print(f"now tok is {input_ids[b_idx][-1].item()}")
            # self.g_cuda.manual_seed(seed)
            ####
            p = torch.softmax(scores[b_idx], dim=-1)
            self.saved_distributions.append(p.detach().cpu().clone())
            filter_indices = top_p_indices(p, self.top_p)
            p_new = torch.zeros(size=(self.vocab_size,), device=p.device)
            p_new[filter_indices] = p[filter_indices]
            ## normalize
            p_new = p_new / p_new.sum()
            ##
            scores[b_idx] = torch.log(p_new+ 1e-10)  # add small value to avoid log(0)
            ####

            bl_ct = int(self.vocab_size*self.bl_proportion)
            self.g_cuda.manual_seed(seed)
            # seed_everything(seed)
            s = torch.randint(low=0,high=2,size=(1,),generator=self.g_cuda,device=input_ids.device)
            self.g_cuda.manual_seed(seed)
            # seed_everything(seed)
            blacklist_ids = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.g_cuda)[:bl_ct] 
            
            side_info.append(s)  #CHANGE TO k

            # print(f's={s.item()}')
            # print(f"list = {blacklist_ids.tolist()[:10]}")

            if self.store_bl_ids: 
                if self.bl_ids is None: self.bl_ids = [[] for _ in range(input_ids.shape[0])]
                self.bl_ids[b_idx].append((blacklist_ids,input_ids.tolist()[b_idx][-1]))

            if self.store_spike_ents: 
                if self.spike_entropies is None: self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self.compute_spike_entropy(scores[b_idx]))
            
            ###
            # pdb.set_trace()
            ###
            
            
            ###
            # The generation process comprises the following steps:
            # 1. generate s (for each element in the batch) and bm (blacklist_ids)
            # 2. Calculate transition kernel A and new logits are l + log(A)
            # 3. save s
            ###

            # self.bad_words_id_length_1[b_idx] = self._prepare_bad_words(blacklist_ids)
            # this logic may not really be necessary for our usecase
            self.bad_words_id_length_1[b_idx] = blacklist_ids

        # concatenate batch side info and store it    
        side_info = torch.concatenate(side_info)
        self.side_info.append(side_info)
        
        
        if not self.noop_blacklist:
            # calc py_tilde -> calc channel_cond -> calc q_tilde logits
            py_tilde = self._calc_py_tilde(scores)
            cc_transform = self._calc_kernel(py_tilde, side_info) 
            scores = self._calc_cc_transform(scores, cc_transform) 

            # pdb.set_trace()
            


            
        # print("scores shape is", scores.shape)
        
        return scores

    def _calc_py_tilde(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        create a batch of py_tilde vectors
        """
        p = scores.softmax(dim=-1)
        self.bad_words_mask = torch.zeros_like(scores)
        py_tilde = []
        for b_idx in range(len(self.bad_words_id_length_1)):
            self.bad_words_mask[b_idx][self.bad_words_id_length_1[b_idx]] = 1.0
            # p1 = torch.sum(p * (1.0 - self.bad_words_mask[b_idx])).reshape(1)
            # py_tilde.append(torch.concatenate([p0,1-p0]))

            p0 = torch.sum(p * (self.bad_words_mask[b_idx])).reshape(1)
            py_tilde = torch.stack([p0,1-p0], -1)
            self.bad_words_mask = self.bad_words_mask.to(dtype=torch.int)
        return py_tilde

    def _calc_kernel(self, p: torch.FloatTensor, side_info: torch.LongTensor) -> torch.FloatTensor:
        """
        calculate the transition kernel A
        """
        # pdb.set_trace()
        A = []
        k = self.k
        channel_cond_j = []
        for b_idx, j in enumerate(side_info):
            # calc p(s=j|tilde{y}) according to optimal coupling
            channel_cond_j.append(self._calc_cond_j(j, p[b_idx],k))
        return channel_cond_j
    
    def _calc_cond_j(self, j: torch.LongTensor, p: torch.FloatTensor, k: int) -> torch.FloatTensor:
        # k = 2
        # Convert j to a Python integer for indexing
        j = int(j.item()) if isinstance(j, torch.Tensor) else int(j)

        # Check if p is exactly uniform: then S = Ytilde w.p.1 so that conditional is an indicator.
        if torch.allclose(p, torch.full_like(p, 1.0 / k)):
            cond = torch.zeros_like(p)
            cond[j] = 1.0
            return cond

        # Total variation distance t (s is uniform)
        t = 0.5 * torch.sum(torch.abs(p - 1.0 / k))

        # A = { i : p[i] >= 1/k } and its complement A^c
        A_mask = p >= (1.0 / k)
        AC_mask = ~A_mask

        # Compute the conditional probabilities row-by-row.
        cond = torch.zeros_like(p)
        for i in range(k):
            xi_row = torch.zeros(k, dtype=p.dtype, device=p.device)
            for l in range(k):
                if i == l:
                    xi_row[l] = torch.min(torch.tensor(1.0 / k, dtype=p.dtype, device=p.device), p[i])
                else:
                    # Only compute off-diagonals if i in A and l in A^c.
                    if A_mask[i] and AC_mask[l]:
                        xi_row[l] = (1.0 / t) * ((1.0 / k - p[i]) * (p[l] - 1.0 / k))
                    else:
                        xi_row[l] = 0.0
            row_sum = torch.sum(xi_row)
            if row_sum > 0:
                cond[i] = xi_row[j] / row_sum
            else:
                cond[i] = 0.0
        return cond
        
    def _calc_cc_transform(self, scores: torch.FloatTensor, cc_transform: torch.FloatTensor) -> torch.FloatTensor:
        """
        apply the channel conditioing transform to the logits
        """
        # print("cc_transform shape is", cc_transform.shape)
        # print("scores shape is", scores.shape)
        k=self.k
        
        for b_idx in range(scores.shape[0]):
            scores[b_idx] = scores[b_idx] + torch.log(k * cc_transform[b_idx][1-self.bad_words_mask[b_idx]])
        return scores 


    def _prepare_bad_words(self, bad_words_ids: List[List[int]]) -> list[int]:
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [self.eos_token_id], bad_words_ids))
        return bad_words_ids
        # used to have more logic, not used now

    def _calc_curr_bad_word_mask(self, scores: torch.FloatTensor) -> torch.BoolTensor:
        bad_words_mask = torch.zeros_like(scores)
        for b_idx in range(len(self.bad_words_id_length_1)):
            bad_words_mask[b_idx][self.bad_words_id_length_1[b_idx]] = 1
        final_mask = bad_words_mask.bool()
        return final_mask

    def _set_scores_to_inf_for_banned_tokens(
        self, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
            # NOTE^^ Omitted logic for dynamic mask based on multi-token ban words
        """
        if self.bl_type == "hard":
            scores = scores.masked_fill(self.bad_words_mask, -float("inf"))
        elif self.bl_type == "soft":
            whitelist_mask = torch.logical_not(self.bad_words_mask)
            blacklist_mask = self.bad_words_mask
            scores[whitelist_mask] = scores[whitelist_mask] + self.bl_logit_bias
            # scores[blacklist_mask] = scores[blacklist_mask] - self.bl_logit_bias  # additive only
        else:
            raise NotImplementedError(f"unrecognized bl type {self.bl_type}!")          
        return scores


def score_sequence(inputs: Tensor = None, 
                   outputs: Tensor = None, 
                   tokenizer: Tokenizer = None,
                #    logits_processor: LogitsProcessor = None,
                   initial_seed: int = None,
                   dynamic_seed: str = None,
                   bl_proportion: float = None,
                   use_cuda: bool = True,
                   record_hits: bool = False,
                   debug: bool = True,
                #    trim_tokens: int = 2,
                ):

    assert  (inputs is not None) and \
            (outputs is not None) and \
            (tokenizer is not None),"output tensor, tokenizer, and bl params req'd"
            # (logits_processor is not None), 

    vocabulary = list(tokenizer.get_vocab().values())
    vocab_size = len(vocabulary)

    model_generations = outputs.tolist()[0] # these are tensors unpack once for speed
    # toks_generated = model_generations[num_orig_input_tokens:]
    toks_generated = model_generations
    num_toks_generated = len(toks_generated)

    # num_toks_to_trim = trim_tokens*2
    # if (num_toks_generated-num_toks_to_trim > 0) == False: 
    #     return -1, -1
        
    # assert num_toks_generated > num_toks_to_trim, f"Need more than {num_toks_to_trim} toks total since we trim start and end a bit."

    # toks_generated = toks_generated[trim_tokens:-trim_tokens]

    if initial_seed is not None:
        random.seed(initial_seed)
    
    device = (torch.device("cuda") if use_cuda else torch.device("cpu"))
    g_cuda = torch.Generator(device=device)
    large_prime = 15485863

    bl_hits, hit_list = 0, []

    prev_token = inputs[0][-1].item()
    # prev_token = toks_generated[0] # haven't decided whether this edge effect matters
    seed_increment = 0
    # for idx,tok_gend in enumerate(toks_generated[1:]):
    for idx,tok_gend in enumerate(toks_generated):

        # prev_token = model_generations[num_orig_input_tokens+idx-1]
        
        if dynamic_seed == "initial":
            g_cuda.manual_seed(large_prime*initial_seed)
        elif dynamic_seed == "markov_1":
            g_cuda.manual_seed(large_prime*prev_token)
        elif dynamic_seed is None:
            # let the rng evolve naturally - this is not a realistic setting
            pass
        elif dynamic_seed == 'fresh':
                seed_increment += 1
                g_cuda.manual_seed(large_prime + seed_increment)


        bl_ct = int(vocab_size*bl_proportion)
        posthoc_blacklist = torch.randperm(vocab_size, device=device, generator=g_cuda)[:bl_ct] # ty Yuxin :]

        tok_in_ph_bl = tok_gend in posthoc_blacklist
        if tok_in_ph_bl:
            bl_hits += 1
            hit_list.append(True)
        else:
            hit_list.append(False)

        if debug: 
            decoded_token = tokenizer.decode(tok_gend, skip_special_tokens=True)
            print(f"Token generated: '{decoded_token}' was in the blacklist {tok_in_ph_bl}")
        
        prev_token = tok_gend

    if debug: 
        print(f"wl hits / num tokens : {num_toks_generated-bl_hits}/{num_toks_generated} = {(num_toks_generated-bl_hits)/num_toks_generated:.02f}")
        print(f"bl hits / num tokens : {bl_hits}/{num_toks_generated} = {bl_hits/num_toks_generated:.02f}")

    if record_hits:
        return bl_hits, num_toks_generated, hit_list
    # bl_fraction = bl_hits/num_toks_generated
    return bl_hits, num_toks_generated


def tokenize_for_generation(example: dict, 
                            idx: int,
                            max_new_tokens: int=None,
                            min_prompt_tokens: int=None,
                            hf_model_name : str=None,
                            tokenizer: Tokenizer=None,
                            model: torch.nn.Module=None):

    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    # preprocess for model generation/completion
    example = tokenize_and_truncate(example, 
                                    completion_length=max_new_tokens, 
                                    prompt_length=min_prompt_tokens,
                                    hf_model_name=hf_model_name,
                                    tokenizer=tokenizer,
                                    # model_max_seq_len=model.config.max_position_embeddings)
                                    model_max_seq_len=None)
    inputs = example["inputs"]
    # for calculating the baseline violation rate across the "gold" completion
    untruncated_inputs = example["untruncated_inputs"]

    # decode the preprocessed input to store for audit
    re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    example.update({"truncated_input":re_decoded_input})

    # also decode the original suffix of the input for audit as the baseline
    decoded_untruncated_input = tokenizer.batch_decode(untruncated_inputs, skip_special_tokens=True)[0]
    example.update({"baseline_completion":decoded_untruncated_input.replace(re_decoded_input,"")})

    example.update({
        "orig_sample_length"            : untruncated_inputs.shape[1],
        "prompt_length"                 : inputs.shape[1],
        "real_completion_length"        : untruncated_inputs.shape[1] - inputs.shape[1],
    })
    return example


def generate_completions(example: dict, 
                        idx: int,
                        max_new_tokens: int=None,
                        hf_model_name : str=None,
                        tokenizer: Tokenizer=None,
                        model: torch.nn.Module=None,
                        no_bl_partial: Callable=None,
                        w_bl_partial: Callable=None,
                        # return_logits: bool=False,
                        bl_processor_list: LogitsProcessorList=None):

    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"


    inputs = example["inputs"]
    re_decoded_input = example["truncated_input"]

    # call the vanilla and watermarked generation function wrappers with the preprocessed inputs
    with torch.no_grad():

        samples_taken = 0
        max_retries = 10
        success = False
        while (success is False) and (samples_taken < max_retries):
            samples_taken += 1
            
            start_generation = time.time()
            outputs_no_bl = no_bl_partial(inputs.to(model.device))
            example["no_bl_gen_time"] = time.time() - start_generation

            # set_seed(1234) # debugging the error when using sampling

            start_generation = time.time()
            outputs_w_bl = w_bl_partial(inputs.to(model.device))
            example["w_bl_gen_time"] = time.time() - start_generation

            # if return_logits:
            #     output_no_bl_dict = outputs_no_bl
            #     logits_no_bl = output_no_bl_dict.scores
            #     outputs_no_bl = output_no_bl_dict.sequences
            #     example["logits_no_bl"] = logits_no_bl

            #     output_w_bl_dict = outputs_w_bl
            #     logits_w_bl = output_w_bl_dict.scores
            #     outputs_w_bl = output_w_bl_dict.sequences
            #     example["logits_w_bl"] = logits_w_bl


            if bl_processor_list:
                if bl_processor_list[0].bl_ids is not None:
                    example["bl_ids"] = bl_processor_list[0].get_and_clear_stored_bl_ids()
                if bl_processor_list[0].spike_entropies is not None:
                    example["spike_entropies"] = bl_processor_list[0].get_and_clear_stored_spike_ents()

            try: 
                # decode and store the new generations for auditing
                no_bl_decoded_output = tokenizer.batch_decode(outputs_no_bl, skip_special_tokens=True)[0]
                example.update({"no_bl_output":no_bl_decoded_output.replace(re_decoded_input,"")})

                w_bl_decoded_output = tokenizer.batch_decode(outputs_w_bl, skip_special_tokens=True)[0]
                example.update({"w_bl_output":w_bl_decoded_output.replace(re_decoded_input,"")})
                
                success = True

            except:
                # log what happened
                print(f"Error while trying to decode the outputs of the model...")
                if samples_taken == 1:
                    print(f"truncated_input: {inputs.tolist()}")
                print(f"Result of attempt {samples_taken}")
                print(f"shape outputs_no_bl: {outputs_no_bl.shape}")
                no_bl_toks = outputs_no_bl.tolist()[0]
                print(f"outputs_no_bl: {no_bl_toks}")
                print(f"outputs_no_bl min: {min(no_bl_toks)}")
                print(f"outputs_no_bl max: {max(no_bl_toks)}")

                print(f"shape outputs_w_bl: {outputs_w_bl.shape}")
                w_bl_toks = outputs_w_bl.tolist()[0]
                print(f"outputs_w_bl: {w_bl_toks}")
                print(f"outputs_w_bl min: {min(w_bl_toks)}")
                print(f"outputs_w_bl max: {max(w_bl_toks)}")
        
        if success is False:
            print(f"Unable to get both a no_bl and w_bl output that were decodeable after {samples_taken} tries, returning empty strings.")
            example.update({"no_bl_output":""})
            example.update({"w_bl_output":""})
            if bl_processor_list:
                if bl_processor_list[0].bl_ids is not None:
                    example["bl_ids"] = []
                if bl_processor_list[0].spike_entropies is not None:
                    example["spike_entropies"] = []

    # Able to get lengths in here by checking 
    # truncated input shape versus the output shape

    example.update({
        # "baseline_num_tokens_generated" : untruncated_inputs.shape[1] - inputs.shape[1], # want this earlier now
        "no_bl_num_tokens_generated"    : outputs_no_bl.shape[1] - inputs.shape[1],
        "w_bl_num_tokens_generated"     : outputs_w_bl.shape[1] - inputs.shape[1]
    })
    example.update({
        "no_bl_sec_per_tok"             : example["no_bl_gen_time"]/example["no_bl_num_tokens_generated"],
        "no_bl_tok_per_sec"             : example["no_bl_num_tokens_generated"]/example["no_bl_gen_time"],
        "w_bl_sec_per_tok"             : example["w_bl_gen_time"]/example["w_bl_num_tokens_generated"],
        "w_bl_tok_per_sec"             : example["w_bl_num_tokens_generated"]/example["w_bl_gen_time"],
    })
    
    # now done externally because these persist outside this func
    # # remove any fields we don't need to keep
    # del example["inputs"]
    # del example["untruncated_inputs"]
    
    return example


def compute_bl_metrics(example: dict, 
                        idx: int,
                        hf_model_name: str = None,
                        tokenizer: Tokenizer=None,
                        initial_seed: int = None,
                        dynamic_seed: str = None,
                        bl_proportion: float = None,
                        use_cuda: bool = None,
                        record_hits: bool = False,
                        limit_output_tokens: int = 0):

    # if example["idx"] == 3: breakpoint()
    # okay need to catch an odd bug here and fix things
    baseline_before = example["baseline_completion"]
    example["baseline_completion"] = baseline_before.replace(example["truncated_input"][:-1],"")
    if example["baseline_completion"] != baseline_before:
        print("baseline input replacement bug occurred!")
    
    no_bl_before = example["no_bl_output"]
    example["no_bl_output"] = no_bl_before.replace(example["truncated_input"][:-1],"")
    if example["no_bl_output"] != no_bl_before:
        print("no_bl_output input replacement bug occurred!")
    
    w_bl_before = example["w_bl_output"]
    example["w_bl_output"] = w_bl_before.replace(example["truncated_input"][:-1],"")
    if example["w_bl_output"] != w_bl_before:
        print("w_bl_output input replacement bug occurred!")

    if ("w_bl_output_attacked" in example):
        w_bl_attacked_before = example["w_bl_output_attacked"]
        example["w_bl_output_attacked"] = w_bl_attacked_before.replace(example["truncated_input"][:-1],"")
        if example["w_bl_output_attacked"] != w_bl_attacked_before:
            print("w_bl_output_attacked input replacement bug occurred!")

    ##########

    # preprocess for model generation/completion
    inputs = tokenize_and_truncate({"text":example["truncated_input"]}, 
                                    completion_length=0, 
                                    hf_model_name=hf_model_name,
                                    tokenizer=tokenizer)["inputs"]

    baseline_outputs = tokenize_and_truncate({"text":example["baseline_completion"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]

    no_bl_outputs = tokenize_and_truncate({"text":example["no_bl_output"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]

    w_bl_outputs = tokenize_and_truncate({"text":example["w_bl_output"]}, 
                                        completion_length=0, 
                                        hf_model_name=hf_model_name,
                                        tokenizer=tokenizer)["inputs"][:,1:]
    if "w_bl_output_attacked" in example:
        w_bl_attacked_outputs = tokenize_and_truncate({"text":example["w_bl_output_attacked"]}, 
                                            completion_length=0, 
                                            hf_model_name=hf_model_name,
                                            tokenizer=tokenizer)["inputs"][:,1:]
    else:
        w_bl_attacked_outputs = None

    if limit_output_tokens > 0:
        example["orig_baseline_completion"] = example["baseline_completion"]
        example["orig_real_completion_length"] = example["real_completion_length"]
        baseline_outputs = baseline_outputs[:,:limit_output_tokens]
        example["real_completion_length"] = baseline_outputs.shape[1]
        example["baseline_completion"] = tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)[0]

        example["orig_no_bl_output"] = example["no_bl_output"]
        example["orig_no_bl_num_tokens_generated"] = example["no_bl_num_tokens_generated"]
        no_bl_outputs = no_bl_outputs[:,:limit_output_tokens]
        example["no_bl_num_tokens_generated"] = no_bl_outputs.shape[1]
        example["no_bl_output"] = tokenizer.batch_decode(no_bl_outputs, skip_special_tokens=True)[0]

        example["orig_w_bl_output"] = example["w_bl_output"]
        example["orig_w_bl_num_tokens_generated"] = example["w_bl_num_tokens_generated"]
        w_bl_outputs = w_bl_outputs[:,:limit_output_tokens]
        example["w_bl_num_tokens_generated"] = w_bl_outputs.shape[1]
        example["w_bl_output"] = tokenizer.batch_decode(w_bl_outputs, skip_special_tokens=True)[0]

        example["orig_spike_entropies"] = example["spike_entropies"]
        example["spike_entropies"] = [example["spike_entropies"][0][:limit_output_tokens]]

        if "w_bl_output_attacked" in example:
            # raise NotImplementedError("Havent thought what to do yet for this")
            example["orig_w_bl_output_attacked"] = example["w_bl_output_attacked"]
            # example["orig_w_bl_attacked_num_tokens_generated"] = example["w_bl_attacked_num_tokens_generated"]
            w_bl_attacked_outputs = w_bl_attacked_outputs[:,:limit_output_tokens]
            example["w_bl_attacked_num_tokens_generated"] = w_bl_attacked_outputs.shape[1]
            example["w_bl_output_attacked"] = tokenizer.batch_decode(w_bl_attacked_outputs, skip_special_tokens=True)[0]

    # score the 3 sequence completions/outputs wrt to bl hits
    result = score_sequence(inputs=inputs,
                            outputs=baseline_outputs, # <-- real text completions
                            initial_seed=initial_seed,
                            dynamic_seed=dynamic_seed,
                            bl_proportion=bl_proportion, 
                            tokenizer=tokenizer,
                            use_cuda=use_cuda,
                            record_hits=record_hits,
                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"baseline_num_toks_gend_eq_0":(num_toks_gend == 0)})
    # if num_toks_gend < 0.99*example["real_completion_length"]: breakpoint()
    # if len(hit_list) < 0.99*example["real_completion_length"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    baseline_stats = {
        "baseline_whitelist_fraction": wl_frac,
        "baseline_blacklist_fraction": bl_frac
    }
    example.update(baseline_stats)
    if record_hits: example.update({"baseline_hit_list":hit_list})
    
    result = score_sequence(inputs=inputs,
                                            outputs=no_bl_outputs, # <-- non-blacklisted version
                                            initial_seed=initial_seed,
                                            dynamic_seed=dynamic_seed,
                                            bl_proportion=bl_proportion, 
                                            tokenizer=tokenizer,
                                            record_hits=record_hits,
                                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"no_bl_num_toks_gend_eq_0":(num_toks_gend == 0)})
    # if num_toks_gend < 0.99*example["no_bl_num_tokens_generated"]: breakpoint()
    # if len(hit_list) < 0.99*example["no_bl_num_tokens_generated"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    no_bl_stats = {
        "no_bl_whitelist_fraction": wl_frac,
        "no_bl_blacklist_fraction": bl_frac
    }
    example.update(no_bl_stats)
    if record_hits: example.update({"no_bl_hit_list":hit_list})
    
    result = score_sequence(inputs=inputs,
                                            outputs=w_bl_outputs, # <-- blacklisted version
                                            initial_seed=initial_seed,
                                            dynamic_seed=dynamic_seed,
                                            bl_proportion=bl_proportion, 
                                            tokenizer=tokenizer,
                                            record_hits=record_hits,
                                            # breakpoint_on_hit=True, # banging head against wall
                                            debug=False)
    if record_hits:
        bl_hits, num_toks_gend, hit_list = result
    else:
        bl_hits, num_toks_gend = result
    example.update({"w_bl_num_toks_gend_eq_0":(num_toks_gend == 0)})
    # if num_toks_gend < 0.99*example["w_bl_num_tokens_generated"]: breakpoint()
    # if len(hit_list) < 0.99*example["w_bl_num_tokens_generated"]: breakpoint()

    if num_toks_gend == 0:
        # print("No tokens generated, odd, avoiding div by zero and returning -1's")
        wl_frac = -1
        bl_frac = -1
    else:
        wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
        bl_frac = bl_hits/num_toks_gend
    w_bl_stats = {
        "w_bl_whitelist_fraction": wl_frac,
        "w_bl_blacklist_fraction": bl_frac
    }
    example.update(w_bl_stats)
    if record_hits: example.update({"w_bl_hit_list":hit_list})

    if w_bl_attacked_outputs is not None:
        result = score_sequence(inputs=inputs,
                                outputs=w_bl_attacked_outputs, # <-- blacklisted but attacked version
                                initial_seed=initial_seed,
                                dynamic_seed=dynamic_seed,
                                bl_proportion=bl_proportion, 
                                tokenizer=tokenizer,
                                record_hits=record_hits,
                                # breakpoint_on_hit=True, # banging head against wall
                                debug=False)
        if record_hits:
            bl_hits, num_toks_gend, hit_list = result
        else:
            bl_hits, num_toks_gend = result
        example.update({"w_bl_attacked_num_toks_gend_eq_0":(num_toks_gend == 0)})
        # if (num_toks_gend-bl_hits)/(num_toks_gend) < 1.0: breakpoint()

        if num_toks_gend == 0:
            # print("No tokens generated, odd, avoiding div by zero and returning -1's")
            wl_frac = -1
            bl_frac = -1
        else:
            wl_frac = (num_toks_gend-bl_hits)/num_toks_gend
            bl_frac = bl_hits/num_toks_gend
        w_bl_attacked_stats = {
            "w_bl_attacked_num_tokens_generated": num_toks_gend,
            "w_bl_attacked_whitelist_fraction": wl_frac,
            "w_bl_attacked_blacklist_fraction": bl_frac
        }
        example.update(w_bl_attacked_stats)
        if record_hits: example.update({"w_bl_attacked_hit_list":hit_list})
    
    return example



def aggregate_bl_stats(example: dict, idx: int, stat_table: dict):
    
    stat_table["baseline_stats"]["whitelist_fraction"] += example["baseline_stats"]["whitelist_fraction"]
    stat_table["baseline_stats"]["blacklist_fraction"] += example["baseline_stats"]["blacklist_fraction"]
    
    stat_table["w_bl_stats"]["whitelist_fraction"] += example["w_bl_stats"]["whitelist_fraction"]
    stat_table["w_bl_stats"]["blacklist_fraction"] += example["w_bl_stats"]["blacklist_fraction"]

    stat_table["no_bl_stats"]["whitelist_fraction"] += example["no_bl_stats"]["whitelist_fraction"]
    stat_table["no_bl_stats"]["blacklist_fraction"] += example["no_bl_stats"]["blacklist_fraction"]

    stat_table["num_examples"] += 1

    return example


def compute_ppl_single(prefix_and_output_text = None,
                        output_text = None,
                        oracle_model_name = None,
                        oracle_model = None,
                        oracle_tokenizer = None):

    with torch.no_grad():
        tokd_prefix = tokenize_and_truncate({"text":prefix_and_output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]
        tokd_inputs = tokd_prefix
        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenize_and_truncate({"text":output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]

        tokd_inputs = tokd_inputs.to(oracle_model.device)
        # make labels, mark if not including all positions
        tokd_labels = tokd_inputs.clone().detach()
        tokd_labels[:,:tokd_labels.shape[1]-tokd_suffix.shape[1]+1] = -100

        outputs = oracle_model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss # avg CE loss all positions (except -100, TODO plz check that this is working correctly)
        ppl = torch.tensor(math.exp(loss))
    
    return loss.item(), ppl.item()


def evaluate_generation_fluency(example: dict, 
                                idx: int,
                                oracle_model_name = None,
                                oracle_model = None,
                                oracle_tokenizer = None):

    # pull out the required fields from the pipeline results
    inputs_plus_baseline_output = f"{example['truncated_input']}{example['baseline_completion']}"
    baseline_output = f"{example['baseline_completion']}"

    inputs_plus_no_bl_output = f"{example['truncated_input']}{example['no_bl_output']}"
    no_bl_output = f"{example['no_bl_output']}"

    inputs_plus_w_bl_output = f"{example['truncated_input']}{example['w_bl_output']}"
    w_bl_output = f"{example['w_bl_output']}"

    # add metrics
    loss, ppl = compute_ppl_single(inputs_plus_baseline_output, baseline_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["baseline_loss"] = loss
    example["baseline_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_no_bl_output, no_bl_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["no_bl_loss"] = loss
    example["no_bl_ppl"] = ppl
    loss, ppl = compute_ppl_single(inputs_plus_w_bl_output, w_bl_output, oracle_model_name, oracle_model, oracle_tokenizer)
    example["w_bl_loss"] = loss
    example["w_bl_ppl"] = ppl

    # del any temp values
    return example


def add_idx(example,idx):
    example.update({"idx":idx})
    return example


def check_input_lengths(example,idx, min_sample_len=0, min_prompt_len=0, min_completion_len=0):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["real_completion_length"]

    # breakpoint()

    conds = all([
        orig_sample_length >= min_sample_len,
        prompt_length >= min_prompt_len,
        real_completion_length >= min_completion_len,
    ])
    return conds


def check_output_lengths(example,min_output_len=0):
    no_bl_output_len = example["no_bl_num_tokens_generated"]
    w_bl_output_len = example["w_bl_num_tokens_generated"]
    conds = all([
        no_bl_output_len >= min_output_len,
        w_bl_output_len >= min_output_len,
    ])
    return conds




class CombinedCCLogitsProcessor(CorrelatedChannelLogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.k = 2
        #initialize bl_ids:
        self.bad_words_id_length_1 = [None for _ in range(input_ids.shape[0])]

        #initialize generator if needed
        # pdb.set_trace()
        if self.g_cuda is None:
            self.g_cuda = torch.Generator(device=input_ids.device)
            print(f"input id device {input_ids.device}")
            # self.g_cuda = torch.Generator(device='cuda')

        side_info = []
        # loop over elements in the batch:
        for b_idx in range(input_ids.shape[0]):
            # pdb.set_trace()
            # set seed:
            if self.dynamic_seed == "initial":
                seed = self.large_prime*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.large_prime*input_ids[b_idx][-1].item()
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.large_prime + self.seed_increment
            
            bl_ct = int(self.vocab_size*self.bl_proportion)
            self.g_cuda.manual_seed(seed)
            # Generate a random permutation of the vocabulary
            permuted_vocab = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.g_cuda)
            # Divide the permuted list into k subsets
            partition = [permuted_vocab[i * bl_ct:(i + 1) * bl_ct] for i in range(self.k - 1)]
            partition.append(permuted_vocab[(self.k - 1) * bl_ct:])  # Add residual elements to the last subset
            
            self.g_cuda.manual_seed(seed)
            # Select the subset corresponding to the current side information
            s = torch.randint(low=0, high=self.k, size=(1,), generator=self.g_cuda, device=input_ids.device)
            blacklist_ids = partition[s.item()]
            side_info.append(s)

            if self.store_bl_ids: 
                if self.bl_ids is None: self.bl_ids = [[] for _ in range(input_ids.shape[0])]
                self.bl_ids[b_idx].append((blacklist_ids, input_ids.tolist()[b_idx][-1]))

            if self.store_spike_ents: 
                if self.spike_entropies is None: self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self.compute_spike_entropy(scores[b_idx]))

            self.bad_words_id_length_1[b_idx] = blacklist_ids

        # concatenate batch side info and store it    
        side_info = torch.concatenate(side_info)
        self.side_info.append(side_info)
        
        # pdb.set_trace()
        if not self.noop_blacklist:
            # apply RG tilting:
            self.bad_words_mask = self._calc_curr_bad_word_mask(scores)
            scores = self._set_scores_to_inf_for_banned_tokens(scores)
            # apply CC tilting:
            py_tilde = self._calc_py_tilde(scores)
            cc_transform = self._calc_kernel(py_tilde, side_info) 
            scores = self._calc_cc_transform(scores, cc_transform) 
                    
        return scores
    

class K_CorrelatedChannelLogitsProcessor(CorrelatedChannelLogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        #initialize bl_ids:
        # pdb.set_trace()
        self.bl_proportion = 1/self.k
        self.bad_words_id_length_1 = [None for _ in range(input_ids.shape[0])]

        #initialize generator if needed
        if self.g_cuda is None:
            self.g_cuda = torch.Generator(device=input_ids.device)
            print(f"input id device {input_ids.device}")

        side_info = []
        # loop over elements in the batch:
        for b_idx in range(input_ids.shape[0]):
            # set seed:
            if self.dynamic_seed == "initial":
                seed = self.large_prime * self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.large_prime * input_ids[b_idx][-1].item()
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.large_prime + self.seed_increment
            # pdb.set_trace()
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

            bl_ct = int(self.vocab_size * self.bl_proportion)
            self.g_cuda.manual_seed(seed)
            s = torch.randint(low=0, high=self.k, size=(1,), generator=self.g_cuda, device=input_ids.device)
            self.g_cuda.manual_seed(seed)
            permuted_vocab = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.g_cuda)
            # Generate a random permutation of the vocabulary
            # qDivide the permuted list into k subsets
            partition = [permuted_vocab[i * bl_ct:(i + 1) * bl_ct] for i in range(self.k - 1)]
            partition.append(permuted_vocab[(self.k - 1) * bl_ct:])  # Add residual elements to the last subset
            
            
            # Select the subset corresponding to the current side information
            # blacklist_ids = partition[s.item()]
            side_info.append(s)

            if self.store_bl_ids: 
                if self.bl_ids is None: self.bl_ids = [[] for _ in range(input_ids.shape[0])]
                self.bl_ids[b_idx].append((blacklist_ids, input_ids.tolist()[b_idx][-1]))

            if self.store_spike_ents: 
                if self.spike_entropies is None: self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self.compute_spike_entropy(scores[b_idx]))

            # self.bad_words_id_length_1[b_idx] = blacklist_ids

        # concatenate batch side info and store it    
        side_info = torch.concatenate(side_info)
        self.side_info.append(side_info)
        
        if not self.noop_blacklist:
            # apply CC tilting:
            py_tilde = self._calc_py_tilde(scores, partition)
            cc_transform = self._calc_kernel(py_tilde, side_info) 
            scores = self._calc_cc_transform(scores, cc_transform)

            # pdb.set_trace()
            if self.tilt:
                scores[0,partition[side_info]] += self.tilting_delta
                    
        return scores
    
    def _calc_py_tilde(self, scores, partition):
        """
        Calculate probabilities for elements in a mask based on the provided list of lists.
        """
        # pdb.set_trace()
        p = scores.softmax(dim=-1)
        py_tilde = []
        self.bad_words_mask = torch.zeros_like(scores)
        for s_idx, id_list in enumerate(partition):
            # Create a mask for the current list of ids
            mask = torch.zeros_like(scores[0], dtype=torch.float)
            mask[id_list] = 1.0
            # Calculate the probability of the elements in the mask
            prob = torch.sum(p[0] * mask).reshape(1)
            py_tilde.append(prob)
            self.bad_words_mask[0][id_list] = s_idx
        # pdb.set_trace()
        self.bad_words_mask = self.bad_words_mask.to(torch.int32)
        return torch.stack(py_tilde,dim=-1)
    
    def _calc_cc_transform(self, scores, cc_transform):
        """
        apply the channel conditioing transform to the logits
        """
        # pdb.set_trace()
        # print("cc_transform shape is", cc_transform.shape)
        # print("scores shape is", scores.shape)
        k=self.k
        
        for b_idx in range(scores.shape[0]):
            scores[b_idx] = scores[b_idx] + torch.log(k * cc_transform[b_idx][self.bad_words_mask[b_idx]])
        return scores
    



class CCWatermarkDetector():
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
        self.seed_increment =0

        if initial_seed is None: 
            self.initial_seed = None
            assert dynamic_seed != "initial"
        else:
            random.seed(initial_seed)
            self.initial_seed = initial_seed
            
        self.dynamic_seed = dynamic_seed
        self.rng = torch.Generator(device=self.device)

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        print(f"green tokens {observed_count} out of {T}")
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

        #if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
        #    print("bos removed!!!")
        #    tokenized_text = tokenized_text[1:]
            
        green_token_count, green_token_mask = 0, []
        
        
        input_sequence = tokenized_text.tolist()[0]
        
        # print("input sequence is:", input_sequence)
        prev_token = inputs[0][-1].item()
        # print("prev_token0 is: ", prev_token)
        
        # prev_token = input_sequence[1]
        # pdb.set_trace()
        cnt_s = 0
        self.seed_increment = 0
        for idx, tok_gend in enumerate(input_sequence):
            if self.dynamic_seed == "initial":
                seed = self.hash_key*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.hash_key*prev_token
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.hash_key + self.seed_increment

            # print(f'seed={seed}')

            redlist_size = int(self.vocab_size*(1 - self.gamma))
            self.rng.manual_seed(seed)
            # seed_everything(seed)
            s = torch.randint(low=0,high=2,size=(1,),generator=self.rng,device=self.device)
            self.rng.manual_seed(seed)
            # seed_everything(seed)
            vocab_permutation = torch.randperm(self.vocab_size, device=self.device,generator=self.rng)
            

            # print(f"s={s.item()}")
            # print(f"list = {vocab_permutation.tolist()[:10]}")
                        
            redlist_ids = vocab_permutation
            if self.select_green_tokens: # directly
                redlist_ids = vocab_permutation[:redlist_size] # new
            else: # select green via red
                redlist_ids = vocab_permutation[(self.vocab_size - redlist_size):]  # legacy behavior
                
            # print(f"len of redlist is {len(redlist_ids) }, the first is {redlist_ids[0]}")    
            
            tok_in_ph_gl = tok_gend in redlist_ids

            # pdb.set_trace()
            # check if s = f(x,bm)
            if s == int(tok_gend in redlist_ids):
            # if s == int(not(tok_gend in redlist_ids)):
                cnt_s += 1
            
            if tok_in_ph_gl:
                green_token_mask.append(False)
                
            else:
                green_token_count += 1
                green_token_mask.append(True)
                
            # if debug:
            #     decoded_token = self.tokenizer.decode(tok_gend, skip_special_tokens=True)
            #     print(f"Token generated: '{decoded_token}' was in the redlist {tok_in_ph_gl}")
            #     print("prev token decode is: ", decoded_token)
            #     print("side info is: ", s)
            prev_token = tok_gend
            
        # pdb.set_trace()
        print("Green Z:")
        z_score = self._compute_z_score(green_token_count, len(input_sequence))
        print("CC Z:")
        z_score_s = self._compute_z_score(cnt_s, len(input_sequence))
        print("z_score is:", z_score)
        print("z_s_score is:", z_score_s)
        return z_score_s


class K_CCWatermarkDetector():
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
                 k: int = 2
                 ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.device = device
        self.tokenizer = tokenizer
        self.select_green_tokens = select_green_tokens
        self.k = k
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
        # count refers to number of green tokens, T is total number of tokens
        # expected_count = self.gamma
        expected_count = 1/self.k
        print(f"green tokens {observed_count} out of {T}")
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

        #if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
        #    print("bos removed!!!")
        #    tokenized_text = tokenized_text[1:]        
        
        input_sequence = tokenized_text.tolist()[0]
        
        # print("input sequence is:", input_sequence)
        prev_token = inputs[0][-1].item()
        # print("prev_token0 is: ", prev_token)
        
        # prev_token = input_sequence[1]
        # pdb.set_trace()
        cnt_s = 0
        self.seed_increment=0
        for idx, tok_gend in enumerate(input_sequence):
            if self.dynamic_seed == "initial":
                seed = self.hash_key*self.initial_seed
            elif self.dynamic_seed == "markov_1":
                seed = self.hash_key*prev_token
            elif self.dynamic_seed == 'fresh':
                self.seed_increment += 1
                seed = self.hash_key + self.seed_increment
            # print(f'seed={seed}')

            bl_ct = int(self.vocab_size/self.k)
            self.rng.manual_seed(seed)
            # seed_everything(seed)
            s = torch.randint(low=0,high=self.k,size=(1,),generator=self.rng,device=self.device)
            self.rng.manual_seed(seed)
            # seed_everything(seed)
            vocab_permutation = torch.randperm(self.vocab_size, device=self.device,generator=self.rng)
            
            # create a mask using partitions:
            #  1) partition into the k sets 
            self.bad_words_mask = torch.zeros_like(vocab_permutation)
            partitions = [vocab_permutation[i * bl_ct:(i + 1) * bl_ct] for i in range(self.k - 1)]
            partitions.append(vocab_permutation[(self.k - 1) * bl_ct:])
            # 2) mask according to the paritions.
            for partition_idx, partition in enumerate(partitions):
                if tok_gend in partition:
                    cnt_s += float(partition_idx == s.item())
            prev_token = tok_gend
            
        # pdb.set_trace()
        print("CC Z:")
        z_score_s = self._compute_z_score(cnt_s, len(input_sequence))
        print("z_s_score is:", z_score_s)
        return z_score_s

