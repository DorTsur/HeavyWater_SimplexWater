from tqdm import tqdm
import os
import torch.nn.functional as F
from watermark.old_watermark import BlacklistLogitsProcessor
from watermark.our_watermark import OurBlacklistLogitsProcessor
# from watermark.gptwm import GPTWatermarkLogitsWarper
# from watermark.watermark_v2 import WatermarkLogitsProcessor
from watermark.cc_watermark import CorrelatedChannelLogitsProcessor, CombinedCCLogitsProcessor, K_CorrelatedChannelLogitsProcessor
from watermark.inverse_transform import InverseTransformLogitsProcessor
from watermark.linear_code import LinearCodeLogitsProcessor
from watermark.gaussian_lincode import GaussianLinearCodeLogitsProcessor
from watermark.exponential import ExponentialLogitsProcessor
from watermark.qarry_linear_code import Q_LinearCodeLogitsProcessor
from watermark.synthid import SynthIDLogitsProcessor
from watermark.heavy_tail_randscore import HeavyTailLogitsProcessor

from transformers import  LogitsProcessorList
import pdb
import random
import torch
import numpy as np
    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

class Generator():
    def __init__(self, args, tokenizer, model) -> None:
        self.mode = args.mode # watermark mode
        self.init_seed, self.dyna_seed, self.gamma, \
        self.delta, self.bl_type, self.num_beams, self.sampling_temp = args.initial_seed, args.dynamic_seed, args.gamma, args.delta, args.bl_type, args.num_beams, args.sampling_temp
        self.tokenizer = tokenizer
        self.model = model # language model
        self.args = args
        
        self.all_token_ids = list(tokenizer.get_vocab().values())
        self.vocab_size = len(self.all_token_ids)
        # if self.vocab_size != model.config.padded_vocab_size:
        #     self.vocab_size = model.config.padded_vocab_size
        
        self.bl_processor = BlacklistLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed,
                                            top_p=self.args.top_p,
                                            temperature=self.sampling_temp,
                                            context=args.context,
                                            hashing=args.hashing_fn,)
        self.logit_processor_lst = LogitsProcessorList([self.bl_processor])
        if args.mode == 'synthid':
            self.bl_processor = SynthIDLogitsProcessor(
                                        vocab_size= self.vocab_size,
                                        dynamic_seed= self.dyna_seed, 
                                        initial_seed= self.init_seed, 
                                        top_p= self.args.top_p,
                                        temperature=self.sampling_temp)
            self.logit_processor_lst = LogitsProcessorList([self.bl_processor])

        if args.mode == 'new': 
            self.bl_processor = OurBlacklistLogitsProcessor(tokenizer=tokenizer,
                                        bad_words_ids=None, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        vocab=self.all_token_ids, 
                                        all_vocab_size=self.vocab_size, 
                                        bl_proportion=1-self.gamma,
                                        bl_logit_bias=self.delta,
                                        bl_type=self.bl_type, 
                                        initial_seed=self.init_seed, 
                                        dynamic_seed=self.dyna_seed)
            self.logit_processor_lst = LogitsProcessorList([self.bl_processor])
            
        if args.mode == 'gpt':
            
            watermark_processor = GPTWatermarkLogitsWarper(vocab_size=self.vocab_size,
                                                        fraction=args.gamma,
                                                        strength=args.delta)
            
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])   
            
        if args.mode == 'v2':
            watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                        gamma=args.gamma,
                                                        delta=args.delta,
                                                        seeding_scheme=args.seeding_scheme,
                                                        select_green_tokens=args.select_green_tokens)
            self.logit_processor_lst = LogitsProcessorList([watermark_processor]) 

        if args.mode == 'cc':
            watermark_processor = CorrelatedChannelLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed,
                                            tilt=args.tilt,
                                            tilting_delta=args.tilting_delta,
                                            top_p=args.top_p
                                            ) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])
        
        if args.mode == 'cc-combined':
            watermark_processor = CombinedCCLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])
        
        if args.mode == 'cc-k':
            watermark_processor = K_CorrelatedChannelLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed,
                                            cc_k=args.cc_k,
                                            tilt=args.tilt,
                                            tilting_delta=args.tilting_delta,
                                            top_p=args.top_p
                                            ) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])
        
        if args.mode == 'inv_tr':
            watermark_processor = InverseTransformLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed
                                            ,top_p=args.top_p
                                            ) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])
        
        if args.mode == 'lin_code':
            watermark_processor = LinearCodeLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed,
                                            tilt=args.tilt,
                                            tilting_delta=args.tilting_delta,
                                            top_p=args.top_p,
                                            context=args.context,
                                            hashing=args.hashing_fn,
                                            temperature = args.sampling_temp,
                                            sinkhorn_reg = args.sinkhorn_reg,
                                            sinkhorn_thresh=args.sinkhorn_thresh
                                            ) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])

        
        if args.mode == 'heavy_tail':
            watermark_processor = HeavyTailLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed,
                                            tilt=args.tilt,
                                            tilting_delta=args.tilting_delta,
                                            top_p=args.top_p,
                                            context=args.context,
                                            hashing=args.hashing_fn,
                                            k=args.heavywater_k,
                                            temperature = args.sampling_temp,
                                            sinkhorn_reg = args.sinkhorn_reg,
                                            sinkhorn_thresh=args.sinkhorn_thresh
                                            ) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])
        

        if args.mode == 'q_lin_code':
            watermark_processor = Q_LinearCodeLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed,
                                            tilt=args.tilt,
                                            tilting_delta=args.tilting_delta,
                                            top_p=args.top_p,
                                            context=args.context,
                                            hashing=args.hashing_fn,
                                            q=args.q
                                            ) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])
        
        if args.mode == 'gauss_lin_code':
            watermark_processor = GaussianLinearCodeLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed,
                                            tilt=args.tilt,
                                            tilting_delta=args.tilting_delta,
                                            top_p=args.top_p,
                                            context=args.context,
                                            hashing=args.hashing_fn
                                            ) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])

        if args.mode == 'exponential':
            watermark_processor = ExponentialLogitsProcessor(
                                            bad_words_ids=None, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            vocab=self.all_token_ids, 
                                            vocab_size=self.vocab_size, 
                                            bl_proportion=1-self.gamma,
                                            bl_logit_bias=self.delta,
                                            bl_type=self.bl_type, 
                                            initial_seed=self.init_seed, 
                                            dynamic_seed=self.dyna_seed,
                                            top_p=args.top_p,
                                            temperature = args.sampling_temp,
                                            ) 
            self.logit_processor_lst = LogitsProcessorList([watermark_processor])
    
        # pdb.set_trace()
    def generate(self, input_ids, max_new_tokens):
        self.logit_processor_lst[0].saved_distributions = []
        self.logit_processor_lst[0].top_p = self.args.top_p

        if self.mode == 'new':
            example = {}
            
            outputs = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                logits_processor = self.logit_processor_lst,
                do_sample=True,
                top_k=0,
                temperature = 1, # temperature applied prior to and implemented in the logit processor
                #temperature=self.sampling_temp,
                top_p= 1 # top-p applied prior to and implemented in the logit processor
            )

            example.update({"bl_vocabularys":self.logit_processor_lst[0].get_and_clear_vocabularys()})
            # remove the attached input from output for some model
            scores = outputs.scores
            output_ids = outputs.sequences[0, -len(scores):]


            # compute logprob for each token
            completions_tokens = []
            completions_logprob = 0

            for score, token, vocabulary in zip(scores, output_ids, example["bl_vocabularys"], strict=True):
                logprobs = F.log_softmax(score[0], dim=-1)
                logprob = logprobs[token].item()
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                    'logprob': logprob,
                    'vocabulary': vocabulary,
                })
                completions_logprob += logprob
            
            completions_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return completions_text, completions_tokens
        else:    
            self.logit_processor_lst[0].seed_increment = 0
            
            if self.mode == 'no':
                
                outputs = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    temperature = self.sampling_temp,
                    top_p= 1 # top-p implemented in the logit processor
                )
            else:    
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    logits_processor = self.logit_processor_lst,
                    do_sample=True,
                    top_k=0,
                    temperature = 1, # temperature applied prior to and implemented in the logit processor
                    #temperature=self.sampling_temp,
                    top_p= 1 # top-p applied prior to and implemented in the logit processor
                )

            

            # elif self.mode == 'old':
            #     # pdb.set_trace()
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp,
            #         top_p= 1 # top-p implemented in the logit processor
            #     )

            # elif self.mode == 'synthid':
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp,
            #         top_p= 1 # top-p implemented in the logit processor
            #     )
            # elif self.mode == 'gpt':
                
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         top_p=0.9
            #     )

            # elif self.mode == 'v2':
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp
            #     )

            # elif self.mode == 'cc' or self.mode == 'cc-combined' or self.mode == 'cc-k':
            #     # pdb.set_trace()
            #     #set seeds:
            #     seed_everything(self.args.initial_seed_llm)
            #     #
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp,
            #         top_p= 1 # top-p implemented in the logit processor
            #     )
            
            # elif self.mode == 'inv_tr':
                
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp,
            #         top_p= 1 # top-p implemented in the logit processor
            #     )

            # elif self.mode == 'lin_code':
            #     seed_everything(self.args.initial_seed_llm)
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp
            #         ,top_p= 1 # top-p implemented in the logit processor
            #     )

            # elif self.mode == 'q_lin_code':
            #     seed_everything(self.args.initial_seed_llm)
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp,
            #         top_p= 1 # top-p implemented in the logit processor
            #     )
            
            # elif self.mode == 'heavy_tail':
            #     seed_everything(self.args.initial_seed_llm)
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp,
            #         top_p= 1 # top-p implemented in the logit processor
            #     )

            # elif self.mode == 'gauss_lin_code':
            #     # pdb.set_trace()
            #     seed_everything(self.args.initial_seed_llm)
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp
            #         ,top_p= 1 # top-p implemented in the logit processor
            #     )

            # elif self.mode == 'exponential':
            #     seed_everything(self.args.initial_seed_llm)
            #     outputs = self.model.generate(
            #         input_ids, max_new_tokens=max_new_tokens,
            #         logits_processor = self.logit_processor_lst,
            #         do_sample=True,
            #         top_k=0,
            #         temperature=self.sampling_temp,
            #         top_p= 1 # top-p implemented in the logit processor
            #     )
            # remove the attached input from output for some model
            # pdb.set_trace()
            scores = outputs.scores
            output_ids = outputs.sequences[0, -len(scores):]
            # access unwatermarked distributions
            # pdb.set_trace()
            if self.mode == 'no':
                original_distributions = scores
            else:
                original_distributions = self.logit_processor_lst[0].saved_distributions  # list of tensors (shape: [vocab_size])

            if self.mode == 'no':
                completions_tokens = []
                for no_watermark_prob, score, token in zip(original_distributions, scores, output_ids, strict=True):
                    completions_tokens.append({
                        'text': self.tokenizer.decode(token),
                        #'CElogprob': CElogprob.item(), #just average this for CE
                    })
                completions_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                return completions_text, completions_tokens, 0

            # print(output_ids)
            # compute logprob for each token
            completions_tokens = []
            completions_logprob = 0
            CE_log_prob_list = []

            #pdb.set_trace()
            # process original distributions for temperature
            for i in range(len(original_distributions)):
                # score = torch.log(original_distributions[i]) / self.sampling_temp
                score = torch.log(original_distributions[i]) 
                original_distributions[i] = torch.softmax(score, dim=-1)
            for no_watermark_prob, score, token in zip(original_distributions, scores, output_ids, strict=True):
                logprobs = F.log_softmax(score[0], dim=-1)
                logprob = logprobs[token].item()
                CElogprob = -np.log(no_watermark_prob[token].item()+1e-8)
                completions_tokens.append({
                    'text': self.tokenizer.decode(token),
                    'logprob': logprob,
                    #'CElogprob': CElogprob.item(), #just average this for CE
                })
                completions_logprob += logprob
                CE_log_prob_list.append(CElogprob.item())
            CE_log_prob_promp = np.mean(CE_log_prob_list)
            print(f"Cross-Entropy is: {CE_log_prob_promp}")
            completions_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return completions_text, completions_tokens, CE_log_prob_promp