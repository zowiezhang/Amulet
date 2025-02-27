import os
import json

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from load_models import load_models

from config import args
from prompts import SYSTEM_PROMPT, PREFERENCE_PROMPTS


class Responser():
    def __init__(self, args, preference=None):
        self.batch_size = args.batch_size
        self.model_name = args.model_name.split('/')[-1]
        self.dataset_name = args.eval_data
        self.args = args
        self.preference = preference
        self.dir_path = os.getcwd()
    
    def get_tookits(self, args):
        tokenizer, model = load_models(args)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer, model
    
    def save_data(self, dataset, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataset = Dataset.from_list(dataset)
        dataset.to_json(file_path, orient='records', lines=False, indent=4)
    
    def load_dataset(self, dataset_name):
        eval_dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")['train']
        # random select 10% data for test
        # eval_dataset = eval_dataset.select(np.random.choice(len(eval_dataset), int(len(eval_dataset) * 0.2)))
        return eval_dataset
    
    def apply_template(self, tokenizer, batch, preference=""):
        
        batch_input = []
        # construct messages
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(preference=preference)},
                {"role": "user", "content": batch[i]['question']}
            ] for i in range(len(batch))
        ]
        for message in messages:
            message_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            batch_input.append(message_text)
        
        batch_inputs = tokenizer(batch_input, padding=True, truncation=True, add_special_tokens=True, max_length=4096, return_tensors="pt")
        return batch_inputs.input_ids

    def apply_template_mistral(self, tokenizer, batch, preference=""):
        batch_input = []
        # construct messages
        batch_input = [SYSTEM_PROMPT.format(preference=preference) + " \n" + batch[i]['question'] for i in range(len(batch))]
        batch_inputs = tokenizer(batch_input, padding=True, truncation=True, add_special_tokens=True, max_length=4096, return_tensors="pt")
        return batch_inputs.input_ids
        
    
    @torch.no_grad()
    def runner(self, subset, args):

        tokenizer, model = self.get_tookits(args)
        indices = np.array_split(np.arange(len(subset)), len(subset)//self.batch_size + 1)
        
        response_list = []
        preference_text = PREFERENCE_PROMPTS[self.preference]
        # get batch responses
        for indice in tqdm(indices):
            batch = subset.select(indice)
            if len(batch) == 0:
                continue
            if ('mistral' in self.model_name) or ('Mistral' in self.model_name):
                batch_base_prompt = self.apply_template_mistral(tokenizer, batch)
                batch_pref_prompt = self.apply_template_mistral(tokenizer, batch, preference=preference_text)
            else:
                batch_base_prompt = self.apply_template(tokenizer, batch)
                batch_pref_prompt = self.apply_template(tokenizer, batch, preference=preference_text)
            batch_base_prompt, batch_pref_prompt = batch_base_prompt.to(args.device), batch_pref_prompt.to(args.device)
            # for various method
            if self.args.method == 'amulet':
                output_ids = model.amulet_generate(tokenizer, batch_base_prompt, batch_pref_prompt, args)
            elif self.args.method == 'la':
                output_ids = model.la_generate(tokenizer, batch_base_prompt, batch_pref_prompt, args)
            elif self.args.method == 'beam':
                output_ids = model.generate(batch_pref_prompt, num_beams = args.num_beams, do_sample = True, early_stopping = True, max_new_tokens = args.max_new_tokens)
                output_ids = output_ids[:, batch_pref_prompt.shape[1]:]
            elif self.args.method == 'pref':
                output_ids = model.generate(batch_pref_prompt, max_new_tokens = args.max_new_tokens)
                output_ids = output_ids[:, batch_pref_prompt.shape[1]:]
            elif self.args.method == 'base':
                output_ids = model.generate(batch_base_prompt, max_new_tokens = args.max_new_tokens)
                output_ids = output_ids[:, batch_base_prompt.shape[1]:]
            
            try:
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            except TypeError:
                outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            
            # process results
            results = [
                {
                    'index': batch[i]['index'],
                    'question': batch[i]['question'],
                    'preference': preference_text,
                    'response': outputs[i]
                } for i in range(len(outputs))
            ]
            response_list.extend(results)

            # optional for reducing the memory cost
            # del outputs
            # del output_ids
            # del batch
            # del batch_base_prompt
            # del batch_pref_prompt
            # torch.cuda.empty_cache()

        return response_list
    
    def get_response(self, dataset_name, preference):
        self.preference = preference
        if self.args.method == 'amulet':
            save_file = f"{self.args.method}{args.iteration_num}"
        elif self.args.method == 'beam':
            save_file = f"{self.args.method}{args.num_beams}"
        else:
            save_file = self.args.method
        self.save_path = f'responses/{self.preference}/{self.model_name}/{dataset_name}/{save_file}.json'
        if os.path.exists(self.save_path):
            print(f"File {save_file}.json already exists!")
            return

        # load dataset
        eval_dataset = self.load_dataset(dataset_name)
        
        results = self.runner(eval_dataset, args)
        results = sorted(results, key=lambda x: x['index'])

        self.save_data(results, os.path.join(self.dir_path, self.save_path))
  













