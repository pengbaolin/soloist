from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange
import json

import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('.')
sys.path.append('./transformers')
sys.path.append('./transformers/')

from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(150)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, token_type_ids, system_token_id, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)

    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    token_type_ids = token_type_ids.unsqueeze(0).repeat(num_samples, 1)
    system_token_id = torch.tensor(system_token_id, dtype=torch.long, device=device)
    system_token_id = system_token_id.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):

            inputs = {'input_ids': generated, 'token_type_ids':token_type_ids}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            token_type_ids = torch.cat((token_type_ids, system_token_id), dim=1)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='gpt2', type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=110)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default='<|endoftext|>', help="Token at which text generation is stopped")
    parser.add_argument('--input_file', type=str, default=None, help="input json file to decoding")
    parser.add_argument('--output_file', type=str, default=None, help="save path")
    parser.add_argument('--max_turn', type=int, default=15, help="number of turns used as context")


    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    inputs = json.load(open(args.input_file))
    output_tests = []
    system_token_id = tokenizer.convert_tokens_to_ids(['system'])
    user_token_id = tokenizer.convert_tokens_to_ids(['user'])

    for idx in range(len(inputs)):
        logger.info(f"PROGRESS: {int(idx/len(inputs)*100)}%")
        example = inputs[idx]
        history = example['history']
        context = history[-args.max_turn:]
        context_ids = []
        token_ids_for_context = []
        for cxt in context:
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
            context_ids += ids
            if 'user :' in cxt:
                token_ids_for_context += user_token_id * len(ids)
            else:
                token_ids_for_context += system_token_id * len(ids)
        
        response = '=>'
        response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))

        context_tokens = context_ids + response_id
        token_type_ids = token_ids_for_context  + system_token_id

        assert( len(context_tokens) == len(token_type_ids))

        out = sample_sequence(
            model=model,
            context=context_tokens,
            token_type_ids=token_type_ids,
            system_token_id=system_token_id,
            num_samples=args.num_samples,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
        )
        out = out[:, len(context_tokens):].tolist()
        examples = []
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find(args.stop_token) if args.stop_token else None]
            examples.append(text)
        
        output_tests.append(examples)
        print(output_tests)
    json.dump(output_tests, open(args.output_file,'w'), indent=2)
    return text


if __name__ == '__main__':
    main()