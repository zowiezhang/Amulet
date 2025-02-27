import random
from copy import deepcopy
from typing import Optional, List

import torch
from transformers.models.qwen2.modeling_qwen2 import *
from transformers import Qwen2ForCausalLM
import torch.nn.functional as F

from utils.top_p_logits import top_p_logits


class DecodingMethodsModel(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, token_size):
        super().__init__(config)
        # self.tokenizer = tokenizer
        self.token_size = token_size

    def amulet_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        del outputs

        return logits

    
    def amulet_score_process(self, tokenizer, logits_player, plain_player_input, player_input, dev, done, args):
        
        logits_player = logits_player[:, -1, :]

        # logits_player = torch.softmax(logits_player.div(args.temperature), dim=-1)
        logits_player = torch.softmax(logits_player, dim=-1)
        probs = top_p_logits(logits_player, topp=args.top_p, filter_value=0)
        
        tok_ids = torch.argmax(probs, dim=-1).to(dev)
        hyp_ids = torch.arange(probs.size(0), device=dev)

        tok_ids = torch.where(done, tokenizer.pad_token_id, tok_ids)

        plain_player_input = torch.cat((plain_player_input, tok_ids.unsqueeze(-1)), dim=-1)
        player_input = torch.cat((player_input, tok_ids.unsqueeze(-1)), dim=-1)

        return plain_player_input, player_input, tok_ids, hyp_ids
    


    @torch.no_grad()
    def la_generate(self, tokenizer, plain_input, pref_input, args, **kwargs):
        
        maxlen_res = args.max_new_tokens
        ratio = args.reinforce_ratio

        dev = plain_input.device
        bsz = plain_input.shape[0]

        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        plain_input = torch.index_select(plain_input, 0, inds)
        pref_input = torch.index_select(pref_input, 0, inds)

        init_length_plain = plain_input.size(1)
        init_length_pref = pref_input.size(1) 

        for _token in range(maxlen_res):

            if done.all():
                break
            
            plain_score = self.amulet_forward(plain_input)
            pref_score = self.amulet_forward(pref_input) 

            pref_score[:, -1, :] = pref_score[:, -1, :] + ratio * (pref_score[:, -1, :] - plain_score[:, -1, :]) 
            
            plain_input, pref_input, tok_ids, hyp_ids = self.amulet_score_process(tokenizer, pref_score, plain_input, pref_input, dev, done, args)
            done = done | tok_ids.eq(tokenizer.eos_token_id)

        # get all finalized candidates for each sample
        plain_input = plain_input[:, init_length_plain:].view(bsz, -1)
        pref_input = pref_input[:, init_length_pref:].view(bsz, -1) 

        return pref_input


    def amulet_game(self, log_ref, log_player, dev, args, **kwargs):

        iter_num = args.iteration_num

        log_player = torch.log(F.softmax(log_player, dim = -1))
        log_ref = torch.log(F.softmax(log_ref, dim = -1))

        bsz = log_player.shape[0]

        Q = torch.zeros((bsz, args.iteration_num + 1, self.token_size), device = dev)
        log_players_0 = deepcopy(log_player)
        log_player_mem = torch.zeros((bsz, args.iteration_num + 1, self.token_size), device = dev)
        log_player_mem[:, 0] = log_players_0[:, -1, :]

        for cur_iter in range(1, iter_num + 1):
            # Update Q_i^{t + 1} at time t
            Q[:, cur_iter] = args.alpha * (log_player[:, -1, :] - log_ref[:, -1, :])

            # Update logits for player_i^{t + 1} 
            log_player[:, -1, :] = \
                    ( args.player_lambda * log_players_0[:, -1, :] + torch.unsqueeze(torch.sum(Q, axis = 1) / cur_iter, dim = 0) + log_player_mem[:,cur_iter-1] / (args.eta * cur_iter) ) / ( args.player_lambda + 1 / (cur_iter * args.eta) )
            
            if cur_iter == iter_num:
                return log_player
                
            log_player = torch.log(F.softmax(log_player, dim = -1))

            # Save current policy into the memory for latter average computation
            log_player_mem[:, cur_iter] = log_player[:, -1, :]
                
        return log_player


    @torch.no_grad()
    def amulet_generate(self, tokenizer, plain_input, pref_input, args, **kwargs):

        maxlen_res = args.max_new_tokens

        dev = plain_input.device
        bsz = plain_input.shape[0]

        done = torch.zeros((bsz,), device = dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        plain_input = torch.index_select(plain_input, 0, inds)
        player_input = torch.index_select(pref_input, 0, inds)

        init_length_player = player_input.size(1) 
        
        for _token in range(maxlen_res):

            if done.all():
                break
            
            logits_ref = self.amulet_forward(plain_input) 
            logits_player = self.amulet_forward(player_input) 

            updated_logits_player = self.amulet_game(logits_ref, logits_player, dev, args)

            plain_input, player_input, tok_ids, hyp_ids = self.amulet_score_process(tokenizer, updated_logits_player, plain_input, player_input, dev, done, args)

            done = done | tok_ids.eq(tokenizer.eos_token_id)
                
                
        # get all finalized candidates for each sample
        player_input = player_input[:, init_length_player:].view(bsz, -1)
        
        # return the last player's result (the player with original prompt)
        return player_input

