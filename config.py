import os
import torch
import argparse

parser = argparse.ArgumentParser(description='Amulet')
parser.add_argument('--method', type=str, default='amulet',
                    help="name of the method, in ['base', 'pref', 'beam', 'la', 'amulet']")
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                    help='path of the using LLM')
parser.add_argument('--eval_data', type=str, default='UltraFeedback_truthful_qa',
                    help='filename of the eval dataset in the datasets folder')
parser.add_argument('--pref_name', type=str, default='creative',
                    help='specific user needs by prompt')
parser.add_argument('--max_new_tokens', type=int, default=128,
                    help='the maximum number of tokens to generate. In other words, the size of the output sequence, not including the tokens in the prompt')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='generation temperature')   
parser.add_argument('--num_processes', type=int, default=1,
                    help='the number of the prallel processes')   
parser.add_argument('--seed', type=int, default=42,
                    help='randomnesss control')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='confidence interval')   
parser.add_argument('--top_k', type=int, default=1,
                    help='decoding top sequences num') 
parser.add_argument('--num_beams', type=int, default=16,
                    help='beam num for beam search')     
parser.add_argument('--eta', type=float, default=10.0,
                    help='learning ratio')   
parser.add_argument('--alpha', type=float, default=2.0,
                    help='reward scale') 
parser.add_argument('--player_lambda', type=float, default=2.0,
                    help='Lagrange multiplier')  
parser.add_argument('--iteration_num', type=int, default=60,
                    help='iteration number for FTRL')  
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')    
parser.add_argument('--reinforce_ratio', type=float, default=2.0,
                    help='reinforcement degree')   
parser.add_argument('--device', type=str, default='cuda:0',
                    help='use which gpu device')
parser.add_argument('--multi_gpu', action='store_true', default=False,
                    help='whether to use multiple GPUs')

args = parser.parse_args()

args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if args.multi_gpu and torch.cuda.is_available():
     args.device = torch.device('cuda')

print("=================Arguments==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

