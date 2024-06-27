# GCG adapted from https://github.com/llm-attacks/llm-attacks

from reconstruction.reconstruct import Reconstructor
import torch
from dataclasses import dataclass
from tqdm import tqdm
import json
import os
from os.path import join
import reconstruction.common as common
import urllib3
import spacy
import pickle
import random
import heapq
import time
import pickle

IGNORE_INDEX = -100


@dataclass
class FullPrompt:
    """
    Holds a single user prompt, documents, suffix
    """

    prompt_ids: list #torch.Tensor
    suffix_slice: list #slice
    # The targets are the docs
    target_prefix_slice: list #slice
    target_prefix_ids: list #torch.Tensor
    prompt_ident: int  # For bookkeeping purposes

    def update_suffix(self, suffix_ids: torch.Tensor) -> None:
        """
        Updates the prompt with a new suffix
        """
        for i in range(len(self.prompt_ids)):
            self.prompt_ids[i] = torch.cat(
                [
                    self.prompt_ids[i][:, : self.suffix_slice[i].start],
                    suffix_ids.unsqueeze(0).to(self.prompt_ids[i].device),
                    self.prompt_ids[i][:, self.suffix_slice[i].stop :], 
                ],
                dim = -1
            )

        # self.prompt_ids = torch.cat(
        #     [
        #         self.prompt_ids[:, : self.suffix_slice.start],
        #         suffix_ids.unsqueeze(0).to(self.prompt_ids.device),
        #         self.prompt_ids[:, self.suffix_slice.stop :],
        #     ],
        #     dim=-1,
        # )


class HardReconstructorGCG(Reconstructor):
    def __init__(
        self,
        num_epochs: int,
        k: int,
        n_proposals: int,
        subset_size: int,
        initial_suffix: str,
        natural_prompt_penalty_gamma: int = 0,  # If 0, no natural prompt penalty
        vocab: str = "",
        warm_start_file: str = "",
        outfile_prefix: str = "niubi",
        start_from_file: str = '',
        reuse_log: str = False,
        autorestart: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.k = k
        self.num_proposals = n_proposals
        self.subset_size = subset_size
        self.natural_prompt_penalty_gamma = natural_prompt_penalty_gamma
        self.vocab = vocab
        self.warm_start_file = warm_start_file
        self.outfile_prefix = outfile_prefix
        self.autorestart = autorestart

        assert initial_suffix, "initial_suffix must be a non empty string"
        try:
            f = open(initial_suffix, 'r')
            self.initial_suffix = f.readline()
        except FileNotFoundError:
            self.initial_suffix = initial_suffix
    
        self.top_suffice = pickle.load(open(start_from_file,'rb')) if start_from_file else []
        if reuse_log and start_from_file: self.outfile_prefix = start_from_file[:-4]
        
        self.vocab_mask = None 
        if self.vocab == 'english':
            words = (
                urllib3.PoolManager()
                .request("GET", "https://www.mit.edu/~ecprice/wordlist.10000")
                .data.decode("utf-8")
            )
            words_list = words.split("\n")
            # nlp = spacy.load("en_core_web_sm")
            # words_list = list(set(nlp.vocab.strings))
            self.vocab_mask = self.get_english_only_mask(words_list)
        elif self.vocab == 'non_english':
            words = (
                urllib3.PoolManager()
                .request("GET", "https://www.mit.edu/~ecprice/wordlist.10000")
                .data.decode("utf-8")
            )
            words_list = words.split("\n")
            # nlp = spacy.load("en_core_web_sm")
            # words_list = list(set(nlp.vocab.strings))
            self.vocab_mask = self.get_non_english_mask(words_list)
        elif self.vocab == 'hybrid':
            words = (
                urllib3.PoolManager()
                .request("GET", "https://www.mit.edu/~ecprice/wordlist.10000")
                .data.decode("utf-8")
            )
            words_list = words.split("\n")
            self.english_mask = self.get_english_only_mask(words_list)
            self.non_english_mask = self.get_non_english_mask(words_list)         
               

    def get_english_only_mask(
        self,
        words_list: list[str],
    ) -> torch.Tensor:
        """
        Get english only tokens from the model's tokenizer
        """

        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size

        english_only_mask = torch.zeros(vocab_size)
        #english_only_mask = torch.ones(vocab_size)
        for word in tqdm(
            words_list, desc="Building non english only mask", total=len(words_list)
        ):
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            for word_id in word_ids:
                english_only_mask[word_id] = 1

        return english_only_mask

    def get_non_english_mask(
        self,
        words_list: list[str],
    ) -> torch.Tensor:
        """
        Get english only tokens from the model's tokenizer
        """

        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size

        #english_only_mask = torch.zeros(vocab_size)
        english_only_mask = torch.ones(vocab_size)
        for word in tqdm(
            words_list, desc="Building non english only mask", total=len(words_list)
        ):
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            for word_id in word_ids:
                english_only_mask[word_id] = 0

        return english_only_mask

    def gcg_gradients(
        self,
        sample: FullPrompt,
    ) -> torch.Tensor:
        """
        First part of GCG. Compute gradients of each suffix (or all) tokens in the sample for Greedy Coordinate Gradient

        Parameters
        ----------
            sample: FullPrompt
                Prompt to compute gradients for

        Returns
        -------
            torch.Tensor
                Gradients of the suffix tokens (suffix_len, vocab_size)
        """

        #assert (
        #    len(sample.prompt_ids.shape) == 2
        #), "prompt_ids must be of shape (1, seq_len)"

        #orig_input_ids = sample.prompt_ids[0].to(self.model.device)
        n_docs = len(sample.target_prefix_ids)
        
        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size

        grads = torch.zeros(
            n_docs,
            sample.suffix_slice[0].stop - sample.suffix_slice[0].start,
            vocab_size,
            device=self.model.device,
        )

        for j in range(0, n_docs, self.batch_size):
            cur_batch_size = min(self.batch_size, n_docs - j)
            embs_list = []
            targets_list = []
            loss_slices = []

            one_hot_suffices = []

            for i in range(j, j + cur_batch_size):

                input_ids = sample.prompt_ids[i].to(self.model.device)
                target_docs = sample.target_prefix_ids[i]#: i + cur_batch_size]
                input_ids = torch.cat([input_ids, target_docs.to(input_ids.device)], dim=1)

                targets = sample.target_prefix_ids[i].to(self.model.device)
                targets_list.append(targets)
                loss_slices.append(slice(
                    sample.target_prefix_slice[i].start - 1,
                    sample.target_prefix_slice[i].stop - 1,
                ))

                model_embeddings = self.model.get_input_embeddings().weight

                one_hot_suffix = torch.zeros(
                    input_ids.shape[0],
                    sample.suffix_slice[i].stop - sample.suffix_slice[i].start,
                    model_embeddings.shape[0],
                    device=self.model.device,
                    dtype=model_embeddings.dtype,
                )

                #print(one_hot_suffix.size())

                one_hot_suffix.scatter_(
                    -1,
                    input_ids[:, sample.suffix_slice[i]].unsqueeze(-1),
                    1,
                )

                one_hot_suffices.append(one_hot_suffix)

                one_hot_suffices[-1].requires_grad = True

                suffix_embs = one_hot_suffices[-1] @ model_embeddings
                embeds = self.model.get_input_embeddings()(input_ids).detach()

                full_embs = torch.cat(
                    [
                        embeds[:, : sample.suffix_slice[i].start, :],
                        suffix_embs,
                        embeds[:, sample.suffix_slice[i].stop :, :],
                    ],
                    dim=1,
                )
                embs_list.append(full_embs)

            embs_list = self.batch_embs(embs_list)
            if 'glm' in self.model.name_or_path:
                logits = self.model(input_ids=torch.ones((embs_list.shape[0], embs_list.shape[1])).to(self.model.device), inputs_embeds=embs_list).logits # temporary workaround before glm fix this (currently it doesnt support inputembed only)
            else:
                logits = self.model(inputs_embeds=embs_list).logits

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

            targets_list = self.batch_targets(targets_list)

            logits = self.batch_logits(logits, loss_slices, targets_list.size(1))
            loss = loss_fct(logits.transpose(1, 2), targets_list)
            # self.accelerator.backward(loss)
            loss.backward()

            for i in range(j, j + cur_batch_size):
                grads[i:i+1] = one_hot_suffices[i - j].grad.clone()

        return grads.mean(dim=0)

    @torch.no_grad()
    def proposal_loss(
        self,
        sample: FullPrompt,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run forward pass with the new proposals and get the loss.

        Parameters
        ----------
            sample: FullPrompt
                Prompt to compute loss for
            proposals: torch.Tensor
                Proposals to compute loss for (num_proposals, prompt_len)

        Returns
        -------
            torch.Tensor
                Loss for each proposal (num_proposals,)
        """

        #print(proposals.shape[0])
        proposal_losses = torch.zeros(
            proposals.shape[0],
            device=proposals.device,
        )
        n_docs = len(sample.target_prefix_ids)
        # Don't think we can batch across the proposal dimensions now since we have many documents (i.e. targets)


        loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="mean", ignore_index=IGNORE_INDEX
                )

        '''
        max_len = max([sample.target_prefix_slice[i].stop -  sample.target_prefix_slice[i].start for i in range(n_docs)] )
        print(max_len)

        all_logits = torch.zeros((proposals.shape[0], n_docs, max_len, self.model.vocab_size)).to(self.model.device)
        all_targets = torch.zeros((proposals.shape[0], n_docs, max_len)).to(sample.target_prefix_ids[0].device).long()

        print(all_logits.size())
        '''
        
        for i in range(proposals.shape[0]):
            #cur_time = time.time()
            proposal = proposals[i] #.repeat(len(sample.target_prefix_ids), 1)
            forward_time = 0
            loss_time = 0
            for k in range(0, n_docs, self.batch_size):
                cur_batch_size = min(self.batch_size, n_docs - k)
                embs_list = []
                #time_1 = time.time()

                for j in range(k, k + cur_batch_size):
                    full_proposal_input = torch.cat(
                        (
                            sample.prompt_ids[j][:, :sample.suffix_slice[j].start],
                            proposal.to(sample.target_prefix_ids[0].device).unsqueeze(0),
                            sample.prompt_ids[j][:, sample.suffix_slice[j].stop:],
                            sample.target_prefix_ids[j],
                        ),
                        dim=1,
                    )

                    #targets = full_proposal_input[:, sample.target_prefix_slice[j]]
                    #targets_list.append(targets)

                    proposal_embs = self.model.get_input_embeddings()(
                        full_proposal_input.to(self.model.device)
                    )
                    embs_list.append(proposal_embs)

                embs_list = self.batch_embs(embs_list)

                if 'glm' in self.model.name_or_path:
                    logits = self.model(input_ids=torch.ones((embs_list.shape[0], embs_list.shape[1])).to(self.model.device), inputs_embeds=embs_list).logits # temporary workaround before glm fix this (currently it doesnt support inputembed only)
                else:
                    logits = self.model(inputs_embeds=embs_list).logits


                #time_2 = time.time()
                for j in range(k, k+cur_batch_size):
                    loss_slice = slice(
                        sample.target_prefix_slice[j].start - 1,
                        sample.target_prefix_slice[j].stop - 1,
                    )

                    proposal_losses[i] += loss_fct(logits[j-k, loss_slice.start:loss_slice.stop].cpu(), sample.target_prefix_ids[j].squeeze() )
                #time_3 = time.time()

                #forward_time += (time_2 - time_1)
                #loss_time += (time_3 - time_2)
                '''
                loss_slices = []
                for j in range(k, k + cur_batch_size):
                    loss_slices.append( slice(
                        sample.target_prefix_slice[j].start - 1,
                        sample.target_prefix_slice[j].stop - 1,
                    ) )

                logits = batch_logits(logits, loss_slices, targets_list.size(1))

                #all_logits[i, k:k+cur_batch_size, :logits.size(1)] = logits
                #all_targets[i, k:k+cur_batch_size, :targets_list.size(1)] = targets_list

                #cur_time = time.time()
                #print(logits.size())

                B0, N0, T0 = logits.size()
                logits = torch.reshape(logits, (B0 * N0, T0))
                target = torch.flatten(targets_list.to(logits.device))
                # #cur_time = time.time()
                loss = loss_fct(
                        logits, target
                        #logits.transpose(1, 2), targets_list.to(logits.device)
                    
                )
                # #print(time.time() - cur_time)

                loss = torch.reshape(loss, (B0, -1))
                # #print("batch backward", str(time.time() - cur_time))
                proposal_losses[i] += loss.sum(dim=0).mean(dim = 0)

                '''
            proposal_losses[i] = proposal_losses[i] / n_docs
            #print(f"total time is {time.time()-cur_time}, forward time is {forward_time}, loss time is {loss_time}")


        return proposal_losses

    def gcg_replace_tok(
        self,
        sample: FullPrompt,
    ) -> tuple[FullPrompt, float]:
        """
        This func implements part 2 of GCG. Now that we have the suffix tokens logits gradients w.r.t loss, we:
        For j = 0,...,total # of proposals
        1. Select the top-k logits for each suffix pos based on -grad of the logits
        2. For each proposal,
        3. Uniformly sample a random token in the top-k logits for replacement at position i
        4. Replace token i with the sampled token. Set this as proposal_j

        Run forward pass for all proposals, get the loss, and pick the proposal with the lowest loss.
        """

        # Compute gradients of the suffix tokens w.r.t the loss
        #cur_time = time.time()
        suffix_logits_grads = self.gcg_gradients(sample)
        #print("gradient time: ", str(time.time() - cur_time))

        #print(suffix_logits_grads.size())
        suffix_logits_grads = suffix_logits_grads / suffix_logits_grads.norm(
            dim=-1, keepdim=True
        )

        if self.vocab in ['english', 'non_english']:
            # clip all non-english tokens
            suffix_logits_grads[:, self.vocab_mask != 1] = float("inf")

            # Select the top-k logits for each suffix pos based on -grad of the logits
            top_k_suffix_logits_grads, top_k_suffix_indices = torch.topk(
                -suffix_logits_grads,
                k=self.k,
                dim=-1,
            )
        elif self.vocab == 'all_allow':
            top_k_suffix_logits_grads, top_k_suffix_indices = torch.topk(
                -suffix_logits_grads,
                k=self.k,
                dim=-1,
            )
        elif self.vocab == 'hybrid':
            suffix_logits_grads_tmp = suffix_logits_grads.detach().clone()
            suffix_logits_grads_tmp[:, self.english_mask != 1] = float("inf")

            # Select the top-k logits for each suffix pos based on -grad of the logits
            top_k_english_suffix_logits_grads, top_k_english_suffix_indices = torch.topk(
                -suffix_logits_grads_tmp,
                k=self.k // 2,
                dim=-1,
            )

            suffix_logits_grads_tmp = suffix_logits_grads.detach().clone()
            suffix_logits_grads_tmp[:, self.non_english_mask != 1] = float("inf")

            # Select the top-k logits for each suffix pos based on -grad of the logits
            top_k_non_english_suffix_logits_grads, top_k_non_english_suffix_indices = torch.topk(
                -suffix_logits_grads_tmp,
                k=self.k // 2,
                dim=-1,
            )
            #print(self.k)
            #print(top_k_english_suffix_logits_grads.size())
            #print(top_k_non_english_suffix_logits_grads.size())
            #print(top_k_english_suffix_indices.size())
            #print(top_k_non_english_suffix_indices.size())
            top_k_suffix_logits_grads = torch.cat([top_k_english_suffix_logits_grads, top_k_non_english_suffix_logits_grads], dim = 1)
            top_k_suffix_indices = torch.cat([top_k_english_suffix_indices, top_k_non_english_suffix_indices], dim = 1)


        self.total_proposals = self.num_proposals * top_k_suffix_indices.shape[0]
        #print(self.num_proposals)
        #print(top_k_suffix_indices.size())
        #j = random.randint(0, len(sample.prompt_ids) - 1)
        #print(sample.prompt_ids[0]
        proposals = sample.prompt_ids[0][:,sample.suffix_slice[0]].repeat(self.total_proposals, 1).to(
            top_k_suffix_indices.device
        )
        #print(proposals)
        '''
        rand_pos = torch.multinomial(
            torch.ones(
                suffix_logits_grads.shape[0],
                device=suffix_logits_grads.device,
            ),
            self.total_proposals,
            replacement=False,
        )
        '''
        for i in range(self.total_proposals):
            proposal_suffix_ids = proposals[i]
            #proposal_suffix_ids = proposal[sample.suffix_slice[j]]
            proposal_suffix_ids[i % suffix_logits_grads.shape[0]] = torch.gather(
                top_k_suffix_indices[i % suffix_logits_grads.shape[0]],
                0,
                torch.randint(
                    0,
                    top_k_suffix_indices.shape[-1],
                    (1,),
                    device=top_k_suffix_indices.device,
                ),
            )
            proposals[i] = proposal_suffix_ids

        # Now compute the loss for each proposal, and pick the next candidate as the lowest one
        with torch.no_grad():
            proposal_losses = self.proposal_loss(sample, proposals)

        best_proposal_idx = proposal_losses.argmin()
        best_proposal = proposals[best_proposal_idx]

        # Now update the sample with the new suffix
        #new_suffix = best_proposal[sample.suffix_slice[j]]
        
        #sample.update_suffix(best_proposal)

        #return sample, proposal_losses.min().item()
        return best_proposal, proposal_losses.min().item()

    def load_datasets(
        self,
        dataset: str | list,
        suffix_only: bool,
        load_doc_tensors: bool = True,
        # start_from_scratch = False,
    ) -> None:
        """
        Load a dataset from a pickle file into the reconstructor

        Parameters
        ----------
            dataset: str | list
                Path to the dataset file or the dataset itself
            suffix_only: bool
                required
            load_doc_tensors: bool
                required
        """

        assert suffix_only, "Reconstruction now always requires suffix"

        if isinstance(dataset, str):
            if load_doc_tensors:
                with open(dataset, "rb") as f:
                    dataset_lst = pickle.load(f)
            else:
                with open(dataset, "r") as f:
                    dataset_lst = json.load(f)
        else:
            dataset_lst = dataset # [{context: [], train_docs_str: []}, {context: , train_docs_str: }]

        data = []
        for d in dataset_lst: #data_lst [[{"conversations": []}, {"objective": str}], [...]]
            # train_docs = []
            # for s in d["train_docs_str"]:
            #     train_docs.append(self.tokenizer.encode(
            #         s,
            #         return_tensors="pt",
            #         add_special_tokens=False,
            #         truncation=True,
            #     )
            # )

            #dev_docs = train_docs[50:]
            #train_docs = train_docs[:50]
            # dev_docs = train_docs

            train_docs = []
            train_prompt_ids = []
            train_suffix_slices = []
            train_target_prefix_slices = []
            dev_prompt_ids = []
            dev_suffix_slices = []
            dev_target_prefix_slices = []

            for entry in d:
                #print(context[i])
                #print(prompt[i])
                #assert 0
                objective_token_id = self.tokenizer.encode(entry['objective'], return_tensors="pt", add_special_tokens=False, truncation=True)
                train_docs.append(objective_token_id)

                context_prompt, train_slice = common.build_context_prompt(
                    self.model.config.name_or_path, entry['conversations'], self.initial_suffix, self.tokenizer
                )
                #prompt_ids.append(context_prompt)
                #all_suffix_slices.append( slice(0, prompt_ids[-1].shape[-1]) )
                #print(self.tokenizer.decode( context_prompt[:, train_slice].squeeze(0) ))
                
                if self.top_suffice: # start_from_file
                    context_prompt[:, train_slice] = heapq.nlargest(1, self.top_suffice)[0][3] # 0 to get the last entry i.e. the best suffix; 3 -> tokens of the suffix
                # elif start_from_scratch:
                #     context_prompt[:, train_slice] = self.tokenizer.encode('!', add_special_tokens=False)[0] # [0] to unpack the list and get the int token
                # otherwise, use the prompt in the pkl

                train_prompt_ids.append(context_prompt)
                train_suffix_slices.append(train_slice)

                train_target_prefix_slices.append(slice(
                    train_prompt_ids[-1].shape[-1],
                    train_prompt_ids[-1].shape[-1] + objective_token_id.shape[-1],
                ) )

                context_prompt, train_slice = common.build_context_prompt(
                    self.model.config.name_or_path, entry['conversations'], self.initial_suffix, self.tokenizer
                )

                #context_prompt[:, train_slice] = 0
                dev_prompt_ids.append(context_prompt)
                dev_suffix_slices.append(train_slice)

                dev_target_prefix_slices.append(slice(
                    dev_prompt_ids[-1].shape[-1],
                    dev_prompt_ids[-1].shape[-1] + objective_token_id.shape[-1],
                ) )   

            #assert 0 
            data.append(
                (
                    FullPrompt(
                        prompt_ids=train_prompt_ids,
                        suffix_slice=train_suffix_slices,
                        target_prefix_slice=train_target_prefix_slices,
                        target_prefix_ids=train_docs,
                        prompt_ident=0,
                    ),
                    FullPrompt(
                        prompt_ids=dev_prompt_ids,
                        suffix_slice=dev_suffix_slices,
                        target_prefix_slice=dev_target_prefix_slices,
                        target_prefix_ids=train_docs,
                        prompt_ident=0,
                    ),
                )
            )

        self.datasets = data

    def train(
        self,
        train_sample: FullPrompt,
        dev_sample: FullPrompt,
        save_path: str,
        prompt_id: int,
        trial: int,
    ) -> dict:
        """
        Optimization for hard prompt reconstruction using GCG

        Parameters
        ----------
            train_sample: FullPrompt
                Prompt to train on
            dev_sample: FullPrompt
                Prompt to evaluate on
            save_path: str
                Path to save results to
            prompt_id: int
                ID of the prompt
            trial: int
                Trial number

        Returns
        -------
            dict
                Dictionary of results w/ prompt id, trial, and the losses/kls
        """

        pbar = tqdm(range(self.num_epochs), total=self.num_epochs)
        best_kl = -float("inf")
        restart_tracker = 0
        # to_ret = []

        kl, std_dev = self.compute_kl(
            prompt1=dev_sample.prompt_ids,
            prompt2=train_sample.prompt_ids,
            docs1=dev_sample.target_prefix_ids,
            docs2=train_sample.target_prefix_ids,
            docs_attn_mask=None,
            p1_attn_mask=None,
            p2_attn_mask=None,
        )


        log_prob_prompt = self.log_prob_prompt_all(
            dev_sample.prompt_ids, dev_sample.suffix_slice
        )
        
        for param in self.model.parameters():
            param.requires_grad = False

        if self.warm_start_file != "" and self.warm_start_file is not None:
            print(
                f"(id {train_sample.prompt_ident}) Original prompt: {self.tokenizer.decode(dev_sample.prompt_ids[0])}"
            )
            print(
                f"(id {train_sample.prompt_ident}) Initial prompt: {self.tokenizer.decode(train_sample.prompt_ids[0])}"
            )
        #print(train_sample.prompt_ids[0])
        #assert 0
        # init_suffix = self.tokenizer.decode(train_sample.prompt_ids[0][0][train_sample.suffix_slice[0]])
        # to_ret.append(
        #     {
        #         "epoch": 0,
        #         "loss": 0,
        #         "kl": kl,
        #         "std_dev": std_dev,
        #         "suffix": init_suffix,
        #         "log_prob_prompt": log_prob_prompt.item(),
        #     }
        # )
        if not self.top_suffice:
            init_id = 0
            
            with torch.no_grad():
                loss_0 = self.proposal_loss(train_sample, train_sample.prompt_ids[0][0][train_sample.suffix_slice[0]].unsqueeze(0).to(self.model.device))

            heapq.heappush(self.top_suffice, (-loss_0, self.initial_suffix, -1, train_sample.prompt_ids[0][0][train_sample.suffix_slice[0]].unsqueeze(0)))

            with open(self.outfile_prefix+".log", "a") as f:
                f.write(f"""
    Initial Prompt: {self.initial_suffix}, 
    Length: {train_sample.suffix_slice[0].stop - train_sample.suffix_slice[0].start} tokens, 
    Loss: {loss_0[0]:.2f}\n""")
                
        else:
            init_id = heapq.nlargest(1, self.top_suffice)[0][2] + 1 # +1 to make sure the new epoch marked from the next id
            with open(self.outfile_prefix+".log", "a") as f:
                f.write(f"""Resume from stored state
    Initial Prompt: {heapq.nlargest(1, self.top_suffice)[0][1]}, 
    Length: {len(heapq.nlargest(1, self.top_suffice)[0][3])} tokens, 
    Loss: {-heapq.nlargest(1, self.top_suffice)[0][0]}\n""")
            
        try:
            for i in pbar:
                
                current_sample = random_choose(train_sample, self.subset_size)

                cur_time = time.time()
                best_proposal, loss = self.gcg_replace_tok(current_sample)
                train_sample.update_suffix(best_proposal)
                suf = self.tokenizer.decode(best_proposal)  # current suffix

                if len(self.top_suffice) < 5:
                    heapq.heappush(self.top_suffice, (-loss, suf, i+init_id, best_proposal)) # minus loss because python heap pq is ascending.
                else:
                    heapq.heappushpop(self.top_suffice, (-loss, suf, i+init_id, best_proposal))
                
                best_loss, _, best_at, _ = heapq.nlargest(1, self.top_suffice)[0]

                with open(self.outfile_prefix+".log", "a", encoding='utf-8') as f:
                    f.write(f"Epoch: {i+init_id}; Suffix: {suf}\nloss:{loss:.2f}; Best KL={best_kl:.2f}; Curr KL={kl:.2f}+-{std_dev:.2f};Logprob. prompt={log_prob_prompt:.2f}\nBest loss so far: {-best_loss} at epoch {best_at}. Average Epoch Speed: {pbar.format_dict['rate']}\n")

                with open(self.outfile_prefix+'.pkl', 'wb') as f:
                    pickle.dump(self.top_suffice, f)

                # if (i + 1) % self.kl_every == 0:
                #     kl, std_dev = self.compute_kl(
                #         prompt1=dev_sample.prompt_ids,
                #         prompt2=train_sample.prompt_ids,
                #         docs1=dev_sample.target_prefix_ids,
                #         docs2=train_sample.target_prefix_ids,
                #         docs_attn_mask=None,
                #         p1_attn_mask=None,
                #         p2_attn_mask=None,
                #     )
                #     log_prob_prompt = self.log_prob_prompt_all(
                #         train_sample.prompt_ids, train_sample.suffix_slice
                #     )
                    
                #     # to_ret.append(
                #     #     {
                #     #         "epoch": i + 1,
                #     #         "loss": loss,
                #     #         "kl": kl,
                #     #         "std_dev": std_dev,
                #     #         "suffix": suf,
                #     #         "log_prob_prompt": log_prob_prompt.item(),
                #     #     }
                #     # )



                # if kl < best_kl:
                #     best_kl = kl
                #     torch.save(
                #         sample.prompt_ids,
                #         join(
                #             save_path,
                #             f"hard_ids_prompt_{prompt_id}_len_{train_sample.target_prefix_ids.shape[-1]}_docs_{train_sample.target_prefix_ids.shape[0]}_trial_{trial}.pt",
                #         ),
                #     )

                pbar.set_description(
                    f"Epoch loss:{loss:.2f};Best loss so far: {-best_loss} at epoch {best_at}."
                )

                if self.autorestart > 0 and (restart_tracker == 0 and i+init_id-best_at > self.autorestart or i+init_id-restart_tracker > self.autorestart and best_at < restart_tracker) :
                    new_start = self.top_suffice[0]
                    restart_tracker = i + init_id
                    train_sample.update_suffix(new_start[3])
                    
                    with open(self.outfile_prefix+".log", "a", encoding='utf-8') as f:
                        f.write(f"Autorestart after not seeing progress in {self.autorestart} epochs. Picked\n{new_start[1]}\nfrom epoch {new_start[2]} with loss {new_start[0]} as the new start.")
                #print(to_ret)
        except KeyboardInterrupt:
            print('I am keyboard interrupted!!!!')
            with open(self.outfile_prefix+'.pkl', 'wb') as f:
                    pickle.dump(self.top_suffice, f)
            print('state saved before exiting.')
            exit(130)

        return {
            "prompt_id": prompt_id,
            "trial": trial,
            # "results": to_ret,
        }


    def batch_embs(self, embs, max_len = None):
        len_list = [embs[i].size(1) for i in range(len(embs))]
        if not max_len:
            max_len = max(len_list)

        for i in range(len(embs)):
            padding = torch.zeros((1, max_len - len_list[i], embs[i].size(2)) ).to(embs[i].device).type(embs[i].dtype)
            embs[i] = torch.cat([embs[i], padding], dim = 1)
        embs = torch.cat(embs, dim = 0)
        return embs

    def batch_targets(self, targets, max_len = None):
        len_list = [targets[i].size(1) for i in range(len(targets))]
        if not max_len:
            max_len = max(len_list)

        for i in range(len(targets)):
            padding = torch.ones((1, max_len - targets[i].size(1))).to(targets[i].device).type(targets[i].dtype) * IGNORE_INDEX
            targets[i] = torch.cat([targets[i], padding], dim = 1)
        targets = torch.cat(targets, dim = 0)
        return targets


    def batch_logits(self, logits, slices, len_targets):
        logits = torch.cat([logits, torch.zeros_like(logits)], dim = 1)
        logits_list = []
        for i in range(len(logits)):
            logits_list.append(logits[i:i+1, slices[i].start:slices[i].start+len_targets])
        logits_list = torch.cat(logits_list, dim = 0)
        return logits_list


def random_choose(sample, subset_size):
    subset_size = min(len(sample.prompt_ids), subset_size)

    idx = list(range(len(sample.prompt_ids)))
    sampled_idx = random.sample(idx, subset_size)

    sampled_idx.sort(key=lambda x: len(sample.prompt_ids[x][0]))

    sampled_prompt_ids = [sample.prompt_ids[i] for i in sampled_idx]
    sampled_suffix = [sample.suffix_slice[i] for i in sampled_idx]
    sampled_prefix = [sample.target_prefix_slice[i] for i in sampled_idx]
    sampled_predix_ids = [sample.target_prefix_ids[i] for i in sampled_idx]

    return FullPrompt(
        prompt_ids=sampled_prompt_ids,
        suffix_slice=sampled_suffix,
        target_prefix_slice=sampled_prefix,
        target_prefix_ids=sampled_predix_ids,
        prompt_ident=0,
    )
