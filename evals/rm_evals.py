import os
import json
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import args


class RMScorer():

    def __init__(self, args, rm_batch_size = 128):
        self.args = args
        self.rm_batch_size = rm_batch_size
        self.dir_path = os.getcwd()
        self.dimen = args.pref_name
        self.model_name = args.model_name
        self.dataset_name = args.eval_data
        rm_path = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
        self.rmodel = AutoModelForSequenceClassification.from_pretrained(rm_path, trust_remote_code = True, torch_dtype = torch.bfloat16).to(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(rm_path, use_fast = True)

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def batch_score(self, messages: List[List[Dict[str, str]]]) -> List[float]:
        batch_input = []
        for message in messages:
            message_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            batch_input.append(message_text)
        
        batch_inputs = self.tokenizer(batch_input, 
                                 padding = True, 
                                 truncation = True, 
                                 add_special_tokens = True, 
                                 max_length = self.args.max_new_tokens, 
                                 return_tensors = "pt")

        input_ids, attention_mask = batch_inputs.input_ids.to(args.device), batch_inputs.attention_mask.to(args.device)
            
        with torch.no_grad():
            output = self.rmodel(input_ids = input_ids, attention_mask = attention_mask)
            # Multi-objective rewards for the response
            multi_obj_rewards = output.rewards.cpu().float()
        
        # return the instruction-following score only
        return multi_obj_rewards[:, 6].tolist()

    def divide_data_batch(self, dataset):
        total_size = len(dataset)
        num_processes = total_size // self.rm_batch_size
        subsets = [dataset[i * self.rm_batch_size : (i + 1) * self.rm_batch_size if i != num_processes - 1 else total_size + 1] for i in range(num_processes)]

        return subsets
    
    def get_rm_eval(self, player_names_list):

        players_list = [self.load_data(os.path.join(self.dir_path, \
                        f"responses/{self.dimen}/{self.model_name}/{self.dataset_name}/{player_name}.json")) \
                        for player_name in player_names_list]

        print(f"Results of {self.dimen}-{self.model_name}-{self.dataset_name} are:")

        for player_idx, player in tqdm(enumerate(players_list)):

            all_batch_rwd = []
            batch_datas = self.divide_data_batch(player)

            for data_subset in tqdm(data_subsets):
                messages = [[
                    {"role": "user", "content": data['question'] + ' ' + data['preference']},
                    {"role": "assistant", "content": data['responses']}
                    ] for data in data_subset]
            
                method_rwd = self.batch_score(messages)
                all_batch_rwd.extend(method_rwd)
                
            avg_rwd = np.array(all_batch_rwd).mean()

            print(f'The average reward for {player_names_list[player_idx]} is {avg_rwd}.')


if __name__ == '__main__':
    evaluator = RMScorer(args)
    player_names_list = ['base', 'pref', 'beam', 'la', 'amulet']
    evaluator.get_rm_eval(player_names_list)
    







