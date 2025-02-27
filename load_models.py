import os

import torch
from transformers import AutoTokenizer


def load_models(args):

    model_last_name = args.model_name.split('/')[-1]
    
    if model_last_name in ['Mistral-7B-Instruct-v0.3', 'Mistral-7B-Instruct-v0.2']:
        from DecodingMethodsModels.MistralDecodingMethods import DecodingMethodsModel
    elif model_last_name in ['Qwen2-0.5B-Instruct', 'Qwen2-7B-Instruct']:
        from DecodingMethodsModels.Qwen2Methods import DecodingMethodsModel
    else:
        from DecodingMethodsModels.LlamaDecodingMethods import DecodingMethodsModel

    print('cuda device_count:', torch.cuda.device_count())
    print('cuda is_available:', torch.cuda.is_available())

    model_name = args.model_name

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, legacy=False)
    print(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    if args.multi_gpu:
        model = DecodingMethodsModel.from_pretrained(
            model_name, token_size = tokenizer.vocab_size, torch_dtype = torch.bfloat16, device_map = 'auto'
        )
    else:
        model = DecodingMethodsModel.from_pretrained(
            model_name, token_size = tokenizer.vocab_size, torch_dtype = torch.bfloat16
        ).half()

    if model.token_size != model.vocab_size:
        model.token_size = model.vocab_size

    if args.multi_gpu:
        model = model.eval()
    else:
        model = model.eval().to(args.device)

    print('Model Loaded!')
    print()

    return tokenizer, model



