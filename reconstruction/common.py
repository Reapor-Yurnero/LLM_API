from cmd import PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import torch
import gc
import multiprocessing as mp
import os
from typing import List, Optional

# SYS_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
SYS_PROMPT = "You are a helpful, respectful and honest assistant. If you don't know the answer to a question, please don't share false information."

PROMPT_TEMPLATES = {
    "vicuna": {
        "prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
        "suffix": "\nASSISTANT:",
    },
    "Llama-2-7b-chat-hf": {
        "prefix": f"[INST] <<SYS>>\n{SYS_PROMPT}\n<</SYS>>\n\n",
        "suffix": " [/INST]",
    },
    "opt": {
        "prefix": "",
        "suffix": "",
    },
    "phi": {
        "prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: ",
        "suffix": "\nASSISTANT:",
    },
    "pythia": {
        "prefix": "",
        "suffix": "",
    },
    "oasst": {
        "prefix": "<|prompter|>",
        "suffix": "<|endoftext|><|assistant|>",
    },
    "openllama": {
        "prefix": "",
        "suffix": "",
    },
    "Llama-3":{
        "prefix": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYS_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    }
}


def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return slice(ind,ind+sll)
        

def prompt_template_handler(model: str, context: List[dict], prompt: str, tokenizer: PreTrainedTokenizer | None, return_tensor='pt') -> List[List[int] | str]:
    prompt = prompt.strip()
    if 'mistral' in model or 'mixtral' in model:
        return llama2_prompt_template(context, prompt, tokenizer, return_tensor)
    
    context = [{'role': 'system', 'content': SYS_PROMPT}] + context
    context.append({'role': 'user', 'content': prompt})

    full_tokens = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=True, return_tensors=return_tensor)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    suffix_slice = find_sub_list(prompt_token_ids, list(full_tokens)[0] if return_tensor != 'pt' else full_tokens.squeeze().tolist()) 
    # [0] since we only have one Conversation so we should always squeeze the output
    assert suffix_slice
    return full_tokens, suffix_slice


def llama3_prompt_template(context: List[dict], prompt: str, tokenizer: PreTrainedTokenizer | None, return_tensor ='pt') -> List[List[int] | str]:
    identity_dict = {'human': 'user',  'gpt': 'assistant', 'system': 'system'}
    # sys prompt and begin oftext
    prompt_str = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYS_PROMPT.strip()}<|eot_id|>"
    # multiturn context
    for conversation in context: 
        prompt_str += '<|start_header_id|>'
        prompt_str += identity_dict[conversation['from']]
        prompt_str += '<|end_header_id|>\n\n'
        prompt_str += conversation['value'].strip()
        prompt_str += '<|eot_id|>'
    # current user response
    prompt_str += '<|start_header_id|>'
    prompt_str += 'user'
    prompt_str += '<|end_header_id|>\n\n'
    # adversarial suffix
    if tokenizer: 
        prompt_ids = tokenizer.encode(prompt_str)
        suffix_start_idx = len(prompt_ids)
    prompt_str += prompt.strip()
    if tokenizer:
        prompt_ids.extend(tokenizer.encode(prompt.strip()))
        suffix_end_idx = len(prompt_ids)
    # assistant start
    prompt_str += '<|eot_id|>'
    prompt_str += '<|start_header_id|>'
    prompt_str += 'assistant'
    prompt_str += '<|end_header_id|>\n\n'
    if tokenizer:
        prompt_ids = tokenizer.encode(prompt_str, return_tensors=return_tensor)
        suffix_slice = slice(suffix_start_idx, suffix_end_idx)
        return prompt_ids, suffix_slice 
    return prompt_str


def llama2_prompt_template(context: List[dict], prompt: str, tokenizer: PreTrainedTokenizer | None, return_tensor ='pt') -> List[List[int] | str]:
    # multiturn context
    prompt_str = ''
    for idx, conversation in enumerate(context): 
        if conversation['from'] == 'human': # user 
            if idx == 0:    # sys prompt and begin of text
                prompt_str += f"<s> [INST] <<SYS>>\n{SYS_PROMPT}\n<</SYS>>\n\n"
            else:
                prompt_str += '<s> [INST] '
            prompt_str += conversation['value'].strip()
            prompt_str += ' [/INST]'
        else: # assistant
            prompt_str += conversation['value'].strip()
            prompt_str += '</s>'
    # current user response
    prompt_str += '<s> [INST] '
    # adversarial suffix
    if tokenizer: 
        prompt_ids = tokenizer.encode(prompt_str)
        suffix_start_idx = len(prompt_ids)
    prompt_str += prompt.strip()
    if tokenizer:
        prompt_ids.extend(tokenizer.encode(prompt.strip()))
        suffix_end_idx = len(prompt_ids)
    # assistant start
    prompt_str += ' [/INST]'
    if tokenizer:
        prompt_ids = tokenizer.encode(prompt_str, return_tensors=return_tensor)
        suffix_slice = slice(suffix_start_idx, suffix_end_idx)
        return prompt_ids, suffix_slice 
    return prompt_str


PROMPTTEMPLATE_HANDLER =  {
    'Llama-3': llama3_prompt_template,
    'Llama-2': llama2_prompt_template,
    'mistral': llama2_prompt_template
    }


MODEL_NAME_OR_PATH_TO_NAME = {
    "lmsys/vicuna-7b-v1.3": "vicuna",
    "/data/models/vicuna/vicuna-7b-v1.5": "vicuna",
    "vicuna": "vicuna",
    "facebook/opt-350m": "opt",
    "facebook/opt-1.3b": "opt",
    "microsoft/phi-1_5": "phi",
    "teknium/Puffin-Phi-v2": "phi",
    "OpenAssistant/oasst-sft-1-pythia-12b": "oasst",
    "EleutherAI/pythia-70m": "pythia",
    "EleutherAI/pythia-160m": "pythia",
    "EleutherAI/pythia-410m": "pythia",
    "EleutherAI/pythia-1b": "pythia",
    "EleutherAI/pythia-1.4b": "pythia",
    "EleutherAI/pythia-2.8b": "pythia",
    "EleutherAI/pythia-6.9b": "pythia",
    "EleutherAI/pythia-12b": "pythia",
    "pythia": "pythia",
    "openlm-research/open_llama_3b_v2": "openllama",
    "/data/models/hf/Llama-2-7b-hf": "openllama",
    "/data/models/hf/Llama-2-7b-chat-hf": "Llama-2-7b-chat-hf",
    "/data/models/hf/Meta-Llama-3-8B-Instruct": "Llama-3",
    "/data/models/hf/Mistral-7B-v0.3": "mistral",
    "/data/models/hf/glm-4-9b-chat": "glm",
    "/data/x5fu/models/hf/Llama-2-7b-chat-hf": "openllama",
    "/data/x5fu/models/hf/Meta-Llama-3-70B-Instruct": "Llama-3"
}

DEVICE_MAPS = {
    "llama_for_causal_lm_2_gpus": {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 0,
        "model.layers.9": 0,
        "model.layers.10": 0,
        "model.layers.11": 0,
        "model.layers.12": 0,
        "model.layers.13": 0,
        "model.layers.14": 0,
        "model.layers.15": 0,
        "model.layers.16": 1,
        "model.layers.17": 1,
        "model.layers.18": 1,
        "model.layers.19": 1,
        "model.layers.20": 1,
        "model.layers.21": 1,
        "model.layers.22": 1,
        "model.layers.23": 1,
        "model.layers.24": 1,
        "model.layers.25": 1,
        "model.layers.26": 1,
        "model.layers.27": 1,
        "model.layers.28": 1,
        "model.layers.29": 1,
        "model.layers.30": 1,
        "model.layers.31": 1,
        "model.norm": 1,
        "lm_head": 1,
    },
    "pythia_2_gpus": {
        "gpt_neox.embed_in": 0,
        "gpt_neox.emb_dropout": 0,
        "gpt_neox.layers.0": 0,
        "gpt_neox.layers.1": 0,
        "gpt_neox.layers.2": 0,
        "gpt_neox.layers.3": 0,
        "gpt_neox.layers.4": 0,
        "gpt_neox.layers.5": 0,
        "gpt_neox.layers.6": 0,
        "gpt_neox.layers.7": 0,
        "gpt_neox.layers.8": 0,
        "gpt_neox.layers.9": 0,
        "gpt_neox.layers.10": 0,
        "gpt_neox.layers.11": 0,
        "gpt_neox.layers.12": 0,
        "gpt_neox.layers.13": 0,
        "gpt_neox.layers.14": 0,
        "gpt_neox.layers.15": 1,
        "gpt_neox.layers.16": 1,
        "gpt_neox.layers.17": 1,
        "gpt_neox.layers.18": 1,
        "gpt_neox.layers.19": 1,
        "gpt_neox.layers.20": 1,
        "gpt_neox.layers.21": 1,
        "gpt_neox.layers.22": 1,
        "gpt_neox.layers.23": 1,
        "gpt_neox.layers.24": 1,
        "gpt_neox.layers.25": 1,
        "gpt_neox.layers.26": 1,
        "gpt_neox.layers.27": 1,
        "gpt_neox.layers.28": 1,
        "gpt_neox.layers.29": 1,
        "gpt_neox.layers.30": 1,
        "gpt_neox.layers.31": 1,
        "gpt_neox.final_layer_norm": 1,
        "embed_out": 1,
    },
    "llama_for_causal_lm_3_gpus": {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 0,
        "model.layers.9": 0,
        "model.layers.10": 0,
        "model.layers.11": 1,
        "model.layers.12": 1,
        "model.layers.13": 1,
        "model.layers.14": 1,
        "model.layers.15": 1,
        "model.layers.16": 1,
        "model.layers.17": 1,
        "model.layers.18": 1,
        "model.layers.19": 1,
        "model.layers.20": 1,
        "model.layers.21": 1,
        "model.layers.22": 2,
        "model.layers.23": 2,
        "model.layers.24": 2,
        "model.layers.25": 2,
        "model.layers.26": 2,
        "model.layers.27": 2,
        "model.layers.28": 2,
        "model.layers.29": 2,
        "model.layers.30": 2,
        "model.layers.31": 2,
        "model.norm": 2,
        "lm_head": 2,
    },
    "llama3_for_causal_lm_3_gpus": {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 0,
        "model.layers.9": 0,
        "model.layers.10": 0,
        "model.layers.11": 0,
        "model.layers.12": 0,
        "model.layers.13": 0,
        "model.layers.14": 0,
        "model.layers.15": 0,
        "model.layers.16": 0,
        "model.layers.17": 0,
        "model.layers.18": 0,
        "model.layers.19": 0,
        "model.layers.20": 0,
        "model.layers.21": 0,
        "model.layers.22": 0,
        "model.layers.23": 0,
        "model.layers.24": 0,
        "model.layers.25": 0,
        "model.layers.26": 0,
        "model.layers.27": 0,
        "model.layers.28": 1,
        "model.layers.29": 1,
        "model.layers.30": 1,
        "model.layers.31": 1,
        "model.layers.32": 1,
        "model.layers.33": 1,
        "model.layers.34": 1,
        "model.layers.35": 1,
        "model.layers.36": 1,
        "model.layers.37": 1,
        "model.layers.38": 1,
        "model.layers.39": 1,
        "model.layers.40": 1,
        "model.layers.41": 1,
        "model.layers.42": 1,
        "model.layers.43": 1,
        "model.layers.44": 1,
        "model.layers.45": 1,
        "model.layers.46": 1,
        "model.layers.47": 1,
        "model.layers.48": 1,
        "model.layers.49": 1,
        "model.layers.50": 1,
        "model.layers.51": 1,
        "model.layers.52": 1,
        "model.layers.53": 1,
        "model.layers.54": 1,
        "model.layers.55": 1,
        "model.layers.56": 1,
        "model.layers.57": 1,
        "model.layers.58": 1,
        "model.layers.59": 2,
        "model.layers.60": 2,
        "model.layers.61": 2,
        "model.layers.62": 2,
        "model.layers.63": 2,
        "model.layers.64": 2,
        "model.layers.65": 2,
        "model.layers.66": 2,
        "model.layers.67": 2,
        "model.layers.68": 2,
        "model.layers.69": 2,
        "model.layers.70": 2,
        "model.layers.71": 2,
        "model.layers.72": 2,
        "model.layers.73": 2,
        "model.layers.74": 2,
        "model.layers.75": 2,
        "model.layers.76": 2,
        "model.layers.77": 2,
        "model.layers.78": 2,
        "model.layers.79": 2,
        "model.norm": 2,
        "lm_head": 2,
    },
}

# not used 
def build_prompt(
    model_name: str, suffix: str, tokenizer: PreTrainedTokenizer
) -> tuple[torch.Tensor, slice]:
    """
    Given the actual "suffix" (prompt), add in the prefix/suffix for the given instruction tuned model

    Parameters
    ----------
        model_name: str
            Model name or path
        suffix: str
            The actual prompt to wrap around
        tokenizer: PreTrainedTokenizer
            Tokenizer for the model

    Returns
    -------
        tuple[torch.Tensor, slice]
            Tuple of the prompt ids and the slice of the actual prompt (suffix)
    """

    model_name = MODEL_NAME_OR_PATH_TO_NAME[model_name]
    cur_prompt = PROMPT_TEMPLATES[model_name]["prefix"]
    suffix_start_idx = max(len(tokenizer(cur_prompt)["input_ids"]) - 1, 0)
    cur_prompt += suffix
    suffix_end_idx = len(tokenizer(cur_prompt)["input_ids"])
    cur_prompt += PROMPT_TEMPLATES[model_name]["suffix"]

    prompt_ids = tokenizer(cur_prompt, return_tensors="pt")["input_ids"]
    suffix_slice = slice(suffix_start_idx, suffix_end_idx)
    return prompt_ids, suffix_slice



def build_context_prompt(
    model_name: str, context: str, prompt: str, tokenizer: PreTrainedTokenizer
) -> tuple[torch.Tensor, slice]:
    """
    Given the actual "suffix" (prompt), add in the prefix/suffix for the given instruction tuned model

    Parameters
    ----------
        model_name: str
            Model name or path
        suffix: str
            The actual prompt to wrap around
        tokenizer: PreTrainedTokenizer
            Tokenizer for the model

    Returns
    -------
        tuple[torch.Tensor, slice]
            Tuple of the prompt ids and the slice of the actual prompt (suffix)
    """

    # model_name = MODEL_NAME_OR_PATH_TO_NAME[model_name]

    if isinstance(context, List):
        return prompt_template_handler(model_name, context, prompt, tokenizer)

    cur_prompt = PROMPT_TEMPLATES[model_name]["prefix"]
    cur_prompt = cur_prompt + context
    #cur_prompt = PROMPT_TEMPLATES[model_name]["prefix"]
    suffix_start_idx = max(len(tokenizer(cur_prompt)["input_ids"]) + 1, 0)
    cur_prompt = cur_prompt + prompt
    suffix_end_idx = len(tokenizer(cur_prompt)["input_ids"])
    cur_prompt = cur_prompt + PROMPT_TEMPLATES[model_name]["suffix"]

    prompt_ids = tokenizer(cur_prompt, return_tensors="pt")["input_ids"]
    suffix_slice = slice(suffix_start_idx, suffix_end_idx)
    return prompt_ids, suffix_slice


def gen_suffix_from_template(
    model_name: str, prompt: str, suffix_char: str, suffix_len: int
) -> tuple[str, str]:
    """
    Given a fully wrapped prompt, replace the actual prompt (suffix) with the control tokens

    Parameters
    ----------
        model_name: str
            Model name or path
        prompt: str
            Prompt to extract suffix from
        suffix_char: str
            Character to use for suffix
        suffix_len: int
            Length of suffix

    Returns
    -------
        tuple[str, str]
            Tuple of the original prompt and the new suffix
    """

    pt = PROMPT_TEMPLATES[MODEL_NAME_OR_PATH_TO_NAME[model_name]]

    orig_prompt = prompt.replace(pt["suffix"], "")
    orig_prompt = orig_prompt.replace(pt["prefix"], "")
    suffix = ((suffix_char + " ") * suffix_len)[:-1]

    return orig_prompt, suffix


def free_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()


def load_model_tokenizer(
    model_name_or_path: str,
    fp16: bool = True,
    device_map = "auto",
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if fp16 else torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    return model, tokenizer


def load_models_tokenizers_parallel(
    model_name_or_path: str,
    fp16: bool = True,
    split_model_gpus: Optional[list[tuple[int, int]]] = None,
) -> tuple[list[AutoModelForCausalLM], list[PreTrainedTokenizer]]:
    """
    Load multiple models for parallel processing **CURRENTLY ONLY SUPPORTS SHARDING ACROSS 2 GPUS**

    Args:
        model_name_or_path (str): Model name or path
        fp16 (bool, optional): Whether to use fp16. Defaults to True.
        split_model (bool, optional): Whether to split the model across multiple (only 2 supported for now) GPUs. Used for hard reconstruction
    """

    models = []
    tokenizers = []

    print("Loading models...")

    if split_model_gpus:
        if 'Llama-3' in MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]:
            dmap = DEVICE_MAPS["llama3_for_causal_lm_3_gpus"]
        elif (
            "vicuna" in MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]
            or "llama" in MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]
            or "Llama" in MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]
        ):
            #dmap = DEVICE_MAPS["llama_for_causal_lm_2_gpus"]
            dmap = DEVICE_MAPS["llama_for_causal_lm_3_gpus"]
        elif "pythia" in MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]:
            dmap = DEVICE_MAPS["pythia_2_gpus"]

        for devices in split_model_gpus:
            if len(devices) == 2:
                device0, device1 = devices
                cur_dmap = {k: device0 if v == 0 else device1 for k, v in dmap.items()}
            elif len(devices) == 3:
                device0, device1, device2 = devices
                cur_dmap = {}
                for k, v in dmap.items():
                    if v == 0:
                        cur_dmap[k] = device0
                    elif v == 1:
                        cur_dmap[k] = device1
                    elif v == 2:
                        cur_dmap[k] = device2
            print(cur_dmap)
            model, tokenizer = load_model_tokenizer(
                model_name_or_path, fp16, device_map=cur_dmap
            )
            models.append(model)
            tokenizers.append(tokenizer)
        # model, tokenizer = load_model_tokenizer(
        #         model_name_or_path, fp16, device_map="auto"
        #     )
        # models.append(model)
        # tokenizers.append(tokenizer)

    else:
        for i in range(torch.cuda.device_count()):
            model, tokenizer = load_model_tokenizer(
                model_name_or_path, fp16, device_map=f"cuda:{i}"
            )
            models.append(model)
            tokenizers.append(tokenizer)

    return models, tokenizers


def setup_multiproc_env(split_models: bool = False):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    mp.set_start_method("spawn")
    n_procs = (
        torch.cuda.device_count() // 2 if split_models else torch.cuda.device_count()
    )
    n_procs = 1 # we added to make sure it doesn't use any multiprocs
    pool = mp.Pool(processes=n_procs)
    return pool


def split_for_multiproc(data: list, n_procs: int) -> list[list]:
    """
    Splits a list into n_procs chunks for multi GPU processing
    """

    n_samples = len(data)
    chunk_size = n_samples // n_procs
    return [data[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
