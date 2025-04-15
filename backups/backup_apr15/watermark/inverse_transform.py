import torch
import pdb
from watermark.old_watermark import BlacklistLogitsProcessor


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
                cdf = torch.cumsum(scores[b_idx, self.permuted_lists[b_idx]], dim=-1)
                # 2. sample uniform [0,1] rv
                u = self.u_samples[b_idx]
                # 3. find CDF index we exceed the threshold
                idx = torch.searchsorted(cdf, u)
                # 4. set that to be the token CDF (set '-inf' to the rest of the scores)
                scores[b_idx, :] = -float('inf')
                scores[b_idx, self.permuted_lists[b_idx][idx]] = 0

        return scores