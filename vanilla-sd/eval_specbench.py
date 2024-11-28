from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.generation.candidate_generator import _crop_past_key_values
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
import argparse, json, time
import pandas as pd
from tqdm.auto import tqdm

LOG_INTERVAL = 50
CATEGORIES = ['mt-bench', 'translation', 'summarization', 'qa', 'math_reasoning', 'rag']

@dataclass
class Args:
    target_model: str
    # when evaluting llm-only generation, pass an empty string as draft_model
    draft_model: str
    experiment_id: str
    test_data_file: str
    output_dir: str
    max_new_tokens: int
    sample: bool = True
    # choose from ['heuristics', 'svip', 'constant']
    length_policy: str = 'constant'
    entropy_threshold: float = 0.3
    draft_length: int = 5
    seed: int = 0
    # the following two arguments are only used when you can't fit the models into one GPU
    memory_first_gpu: int = 20
    memory_per_gpu: int = 36

    def dict(self):
        return asdict(self)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    arg = parser.parse_args("--config config.json".split())
    with open(arg.config) as f:
        config = json.load(f)
    args = Args(**config)
    assert args.length_policy in ['heuristics', 'svip', 'constant'], "invalid adaptive mode"
    return args


args = parse_args()

print(args)
set_seed(args.seed)
device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained(args.target_model)
if torch.cuda.device_count() == 1:
    model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.bfloat16).to(device)
else:
    max_memory = {0: f"{args.memory_first_gpu}GiB", **{i: f"{args.memory_per_gpu}GiB" for i in range(1, torch.cuda.device_count())}}
    model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.bfloat16, device_map="auto", max_memory=max_memory)

model.resize_token_embeddings(len(tokenizer))

df = pd.read_json(args.test_data_file, lines=True)
prompt_tokenized = [tokenizer(p, return_tensors='pt').input_ids.to(device) for p in df.prompt]

@torch.inference_mode()
def generate(model, input_ids, past_key_values=None, max_new_tokens=512):
    new_token_count = 0
    t1 = time.time()
    pbar = tqdm(total=max_new_tokens)
    while True:
        # logits: (1, seqlen, vocab)
        model_inputs = input_ids if past_key_values is None else input_ids[:,-1:]
        outputs = model(model_inputs, past_key_values=past_key_values, return_dict=True)
        
        next_token_logits = outputs.logits[:, -1, :]
        if args.sample:
            p = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(p, 1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]

        input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        past_key_values = outputs.past_key_values

        new_token_count += 1

        if new_token_count % LOG_INTERVAL == 0:
            pbar.update(LOG_INTERVAL)
        
        if new_token_count >= max_new_tokens or next_tokens.item() == tokenizer.eos_token_id:
            break
    pbar.close()
    return input_ids, time.time() - t1

# no SD
if not args.draft_model:
    outputs, outputs_ids, num_tokens, times = [], [], [], []
    for input_ids in tqdm(prompt_tokenized):
        output_ids, time_log = generate(model, input_ids, max_new_tokens=args.max_new_tokens)
        outputs.append(tokenizer.decode(output_ids[0], skip_special_tokens=False))
        outputs_ids.append(list(output_ids[0].cpu().numpy()))
        num_tokens.append(output_ids.shape[1]-input_ids.shape[1])
        times.append(time_log)
    df['generation'] = outputs
    df['output_ids'] = outputs_ids
    df['num_tokens'] = num_tokens
    df['time'] = times
    df.to_json(f"{args.output_dir}/{args.experiment_id}.jsonl", orient='records', lines=True)
    df['speed'] = [n/t for n, t in zip(num_tokens, times)]
    print('Speed:')
    for cat in CATEGORIES:
        df_ = df[df.category==cat]
        print(f"{df_.speed.mean():.2f}", end='\t')
    print(f"{df.speed.mean():.2f}")
    exit()


draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype=torch.bfloat16).to(device)
draft_model.resize_token_embeddings(len(tokenizer))

@torch.inference_mode()
def calc_entropy(logits):
    # logits: (seqlen, vocab)
    assert len(logits.shape) == 2
    logits = logits - logits.max(dim=1).values.unsqueeze(1)
    exp_logits = torch.exp(logits)
    return -((logits - torch.log(exp_logits.sum(dim=1).unsqueeze(1))) * exp_logits / exp_logits.sum(dim=1).unsqueeze(1)).sum(dim=1)

class AssistantModel:
    def __init__(self, model):
        self.model = model
        self.past_key_values = None
        self.entropy_threshold = args.entropy_threshold
    
    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens == 0:
            return input_ids, None
        max_new_tokens = min(40, max_new_tokens)
        new_token_count = 0
        candidate_probs = []
        while True:
            # logits: (1, seqlen, vocab)
            uncached_tokens = input_ids.shape[1] - self.past_key_values[0][0].shape[2]
            model_inputs = input_ids[:,-uncached_tokens:]
            outputs = self.model(model_inputs, past_key_values=self.past_key_values, return_dict=True)
            
            # (1, vocab)
            next_token_logits = outputs.logits[:, -1, :]
            entropy = calc_entropy(next_token_logits) if args.length_policy == 'svip' else None

            if args.sample:
                p = F.softmax(next_token_logits, dim=-1)
                candidate_probs.append(p)
                next_tokens = torch.multinomial(p, 1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            self.past_key_values = outputs.past_key_values
            
            new_token_count += 1
            if new_token_count >= max_new_tokens or next_tokens.item() == tokenizer.eos_token_id:
                break
            if args.length_policy == 'svip' and torch.sqrt(entropy * 0.15) > self.entropy_threshold:
                break
        # stack logits to (bs, num_draft_tokens, vocab) than squeeze the bs dimension
        candidate_probs = torch.stack(candidate_probs, dim=1)[0] if candidate_probs else []
        return input_ids, candidate_probs
    
def assisted_decoding(model, assistant_model, input_ids, past_key_values=None, max_new_tokens=512, num_assistant_tokens=5):
    if args.length_policy not in ['heuristics', 'constant']:
        num_assistant_tokens = 50000
    # only support batch size = 1
    candidate_generator = AssistantModel(assistant_model)
    new_token_count = 0
    step, generated_draft_tokens, accepted_draft_tokens = 0, 0, 0
    last_input_id_index, correct_draft_tokens = [], []
    pbar = tqdm(total=max_new_tokens)

    t1 = time.time()
    # get kvcache for prompt
    # leave out the last token to avoid input of shape [1,0] in the first draft process and other issues
    outputs = model(input_ids[:,:-1], return_dict=True)
    past_key_values = outputs.past_key_values
    outputs = assistant_model(input_ids[:,:-1], return_dict=True)
    candidate_generator.past_key_values = outputs.past_key_values
    
    while True:
        cur_len = input_ids.shape[-1]
        #  1. Fetch candidate sequences
        max_assistant_tokens = min(num_assistant_tokens, max_new_tokens - new_token_count - 1)
        candidate_input_ids, candidate_probs = candidate_generator.generate(input_ids, max_new_tokens=max_assistant_tokens)
        last_input_id_index.append(input_ids.shape[1]-1)

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        candidate_new_tokens = candidate_input_ids[:, cur_len:] # (bs, cand_len)

        # 2. Verification
        model_inputs = candidate_input_ids[:,-candidate_length-1:]
        outputs = model(model_inputs, past_key_values=past_key_values, return_dict=True)
        assert outputs.logits.shape[1] == candidate_length + 1

        # We can keep the candidate tokens until the first mismatch, or until the max length is reached.
        if args.sample and candidate_length > 0:
            # p: target, q: draft
            target_probs = F.softmax(outputs.logits, dim=-1)[0]         # (cand_len+1, vocab)
            q = candidate_probs.gather(1, candidate_new_tokens.view(-1, 1)).view(1, -1) # (1, cand_len)
            p = target_probs.gather(1, candidate_new_tokens.view(-1, 1)).view(1, -1).to(device)
            r = torch.rand(p.shape, device=p.device)
            n_matches = int(((~(r < p/q)).cumsum(dim=-1) < 1).sum())
            valid_tokens = candidate_new_tokens[:, :n_matches]
            if n_matches < candidate_length:
                # sample from norm(p-q) at position n_matches
                diff_pos = F.relu(target_probs[n_matches, :].to(device) - candidate_probs[n_matches, :])
                p_new = diff_pos / diff_pos.sum()
                valid_tokens = torch.cat([
                    valid_tokens,
                    torch.multinomial(p_new.unsqueeze(0), 1)
                ], dim=-1)
            else:
                # sample from last target_probs
                valid_tokens = torch.cat([
                    valid_tokens,
                    torch.multinomial(target_probs[-1:, :].to(device), 1)
                ], dim=-1)
        else:
            # greedy verification
            selected_tokens = outputs.logits.argmax(dim=-1)     # (1, cand_len+1)
            n_matches = int(((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum())
            valid_tokens = selected_tokens[:, : n_matches + 1].to(device)

        if candidate_input_ids[0][-1].item() == tokenizer.eos_token_id and n_matches == candidate_length:
            valid_tokens = valid_tokens[:, :-1]
        correct_draft_tokens.append(n_matches)

        # update input ids and kv cache
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        new_cur_len = input_ids.shape[-1]
        new_cache_size = new_cur_len - 1
        # correct cache size of target model is always new_cur_len - 1
        # that of draft model is new_cur_len - 1 if at least one draft token is invalide, and otherwise new_cur_len - 2
        # but in the latter case, the draft model only has new_cur_len - 2 cache in total
        past_key_values = _crop_past_key_values(model, outputs.past_key_values, new_cache_size)
        candidate_generator.past_key_values = _crop_past_key_values(
            candidate_generator.model, candidate_generator.past_key_values, new_cache_size
        )
            
        # at this point if all draft models are correct, draft kv cache will 
        # be one token shorter than target kv cache. Otherwise they will have the same length

        # Update the candidate generation strategy if needed
        if args.length_policy == 'heuristics':
            if n_matches == num_assistant_tokens:
                num_assistant_tokens += 2
                num_assistant_tokens = min(num_assistant_tokens, 40)
            else:
                num_assistant_tokens = max(1, num_assistant_tokens - 1)

        new_token_count += (n_matches + 1)
        step += 1
        generated_draft_tokens += candidate_length
        accepted_draft_tokens += n_matches
        pbar.update(n_matches + 1)
        if new_token_count >= max_new_tokens or input_ids[0][-1].item() == tokenizer.eos_token_id:
            break
    pbar.close()

    return input_ids, new_token_count, accepted_draft_tokens, step, generated_draft_tokens, last_input_id_index, correct_draft_tokens, time.time() - t1


outputs, new_token_counts, accepted_token_counts, steps, draft_token_counts, outputs_ids, num_tokens, times = [], [], [], [], [], [], [], []
last_id_index, n_match = [], []
speeds = []
for input_ids in tqdm(prompt_tokenized):
    with torch.inference_mode():
        output_ids, new_token_count, accepted_draft_tokens, step, generated_draft_tokens, last_input_id_index, correct_draft_tokens, t = assisted_decoding(model, draft_model, input_ids, max_new_tokens=args.max_new_tokens, num_assistant_tokens=args.draft_length)

    outputs.append(tokenizer.decode(output_ids[0], skip_special_tokens=False))
    new_token_counts.append(new_token_count)
    accepted_token_counts.append(accepted_draft_tokens)
    steps.append(step)
    draft_token_counts.append(generated_draft_tokens)
    last_id_index.append(last_input_id_index)
    n_match.append(correct_draft_tokens)
    outputs_ids.append(list(output_ids[0].cpu().numpy()))
    num_tokens.append(output_ids.shape[1]-input_ids.shape[1])
    times.append(t)

df['generation'] = outputs
df['new_token_count'] = new_token_counts
df['generation_steps'] = steps
df['draft_token_count'] = draft_token_counts
df['accepted_token_count'] = accepted_token_counts
df['last_id_index'] = last_id_index
df['n_match'] = n_match
df['output_ids'] = outputs_ids
df['num_tokens'] = num_tokens
df['time'] = times
df.to_json(f"{args.output_dir}/{args.experiment_id}.jsonl", orient='records', lines=True)
df['speed'] = [n/t for n, t in zip(num_tokens, times)]
print('Speed:')
for cat in CATEGORIES:
    df_ = df[df.category==cat]
    print(f"{df_.speed.mean():.2f}", end='\t')
print(f"{df.speed.mean():.2f}")
