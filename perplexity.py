from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm

def calc_perplexity(model_type, file):
    f = open(file, "r")
    text = f.read()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", eos_token='<|endoftext|>')

    encodings = tokenizer(text, return_tensors='pt')
    model = GPT2LMHeadModel.from_pretrained(model_type)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_type)

    model = GPT2LMHeadModel.from_pretrained("./data/models" + model_type)

    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl