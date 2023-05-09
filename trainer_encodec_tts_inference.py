import json
from argparse import ArgumentParser, Namespace

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from encodec_model.nar_bart_model import NARBartForConditionalGeneration


def inference_ar(ar_model, ar_tokenizer, dataset, device, batch=1):
    decoder_outputs = {}
    force_words_ids = ar_tokenizer([f"v_tok_{u}" for u in range(1024)], 
                                  add_special_tokens=True).input_ids
    for i in tqdm(range(len(dataset))):
        file_id = dataset['id'][i]
        inputs = ar_tokenizer(dataset['text'][i], padding='max_length', truncation=True,
                              max_length=1024, return_tensors="pt").to(device)
        output_ids = ar_model.generate(input_ids=inputs['input_ids'], num_beams=1, do_sample=False,
                                       max_length=1024)#, force_words_ids=force_words_ids)
        decode_output = ar_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        decoder_outputs[file_id] = [int(token.strip(' ')) for token in decode_output.split('v_tok_')[1:]]
    return decoder_outputs


def filter_token(output, target_tokens):
    for i in range(output.logits.size(-1)):
        if i not in target_tokens:
            output.logits[:, :, i] = float('-inf')
    return output


def get_nar_target_token_sets(tokenizer):
    target_token_sets = [
        ' '.join([f"v_tok_{u + l * 1024}" for u in range(1024)])
        for l in range(1, 8)
    ]
    target_token_sets = [
        set(tokenizer.encode(tokens, add_special_tokens=False))
        for tokens in target_token_sets
    ]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and eos_token_id not in target_token_sets:
        target_token_sets = [token_set.union({eos_token_id}) for token_set in target_token_sets]
    
    return target_token_sets


def inference_nar(nar_model, nar_tokenizer, dataset, device, use_gt=False):
    decoder_outputs = {}
    target_token_sets = get_nar_target_token_sets(nar_tokenizer)
    
    for i in tqdm(range(len(dataset))):
        file_id = dataset['id'][i]
        inputs = nar_tokenizer([dataset['text'][i]], padding='max_length', truncation=True, 
                               max_length=1024, return_tensors="pt").to(device)
        decoder_outputs[file_id] = decoder_outputs.get(file_id, {})
        
        for l in range(7):
            if l == 0 or use_gt:
                decoder_input_ids = nar_tokenizer.convert_tokens_to_ids(
                    [f"v_tok_{u + l * 1024}" for u in dataset[f'encodec_{l}'][i]])
            else:
                decoder_input_ids = nar_tokenizer.convert_tokens_to_ids(
                    [f"v_tok_{u + l * 1024}" for u in decoder_outputs[file_id][f'encodec_{l}']])
            decoder_input_ids = torch.tensor([decoder_input_ids], device=device)
            output = nar_model(inputs['input_ids'], decoder_input_ids=decoder_input_ids)
            output = filter_token(output, target_token_sets[l])
            decode_ids = torch.argmax(output.logits, dim=-1)
            decode_output = nar_tokenizer.batch_decode(decode_ids, skip_special_tokens=True)

            decoder_outputs[file_id][f'encodec_{l + 1}'] = \
                [int(token.strip(' ')) - (l + 1) * 1024 for token in decode_output[0].split('v_tok_')[1:]]
    
    return decoder_outputs


def inference_ar_nar(ar_model, ar_tokenizer, nar_model, nar_tokenizer, dataset, device):
    decoder_outputs = {}
    target_token_sets = get_nar_target_token_sets(nar_tokenizer)
    
    for i in tqdm(range(len(dataset))):
        file_id = dataset['id'][i]
        decoder_outputs[file_id] = decoder_outputs.get(file_id, {})
        
        inputs = ar_tokenizer(dataset['text'][i], padding='max_length', truncation=True,
                              max_length=1024, return_tensors="pt").to(device)
        ar_output_ids = ar_model.generate(input_ids=inputs['input_ids'], num_beams=1, do_sample=False,
                                          max_length=1024)
        ar_decode_output = ar_tokenizer.decode(ar_output_ids[0], skip_special_tokens=True)
        decoder_outputs[file_id]["encodec_0"] = [int(token.strip(' ')) for token in ar_decode_output.split('v_tok_')[1:]]
        
        for l in range(1, 8):
            decoder_input_ids = nar_tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (l - 1) * 1024}" for u in decoder_outputs[file_id][f"encodec_{l - 1}"]])
            decoder_input_ids = torch.tensor([decoder_input_ids], device=device)
            output = nar_model(inputs['input_ids'], decoder_input_ids=decoder_input_ids)
            output = filter_token(output, target_token_sets[l - 1])
            decode_ids = torch.argmax(output.logits, dim=-1)
            decode_output = nar_tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
            
            decoder_outputs[file_id][f"encodec_{l}"] = \
                [int(token.strip(' ')) - l * 1024 for token in decode_output[0].split('v_tok_')[1:]]
    
    return decoder_outputs
        


def write_decoder_output_ids(ar_outputs, nar_outputs, result_json):
    assert set(ar_outputs.keys()) == set(nar_outputs.keys())
    
    result = {
        k: {**{f"encodec_0": ar_outputs[k]}, **nar_outputs[k]}
        for k in ar_outputs.keys()
    }
    with open(result_json, 'w+') as f:
        json.dump(result, f)


def main(args):
    device = args.device
    
    # dataset
    dataset = load_dataset("voidful/librispeech_encodec", split="validationclean")
    dataset = dataset.filter(lambda x : len(x[f"encodec_0"]) <= 1000)
    dataset = dataset.shuffle(seed=42).select(range(30))
    
    if args.cascade_ar_nar:
        ar_model = eval(args.ar_model).from_pretrained(args.ar_model_ckpt).to(device)
        ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_model_ckpt)
        nar_model = eval(args.nar_model).from_pretrained(args.nar_model_ckpt).to(device)
        nar_tokenizer = AutoTokenizer.from_pretrained(args.nar_model_ckpt)
        decoder_output_ids = inference_ar_nar(ar_model, ar_tokenizer, nar_model, nar_tokenizer, dataset, device)
        with open(args.result_json, 'w+') as f:
            json.dump(decoder_output_ids, f)
    else:
        if args.use_ar_model:
            ar_model = eval(args.ar_model).from_pretrained(args.ar_model_ckpt).to(device)
            ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_model_ckpt)
            ar_decoder_output_ids = inference_ar(ar_model, ar_tokenizer, dataset, device)
        else:
            ar_decoder_output_ids = {
                dataset['id'][i]: dataset['encodec_0'][i]
                for i in range(len(dataset))
            }

        if args.use_nar_model:
            nar_model = eval(args.nar_model).from_pretrained(args.nar_model_ckpt).to(device)
            nar_tokenizer = AutoTokenizer.from_pretrained(args.nar_model_ckpt)
            nar_decoder_output_ids = inference_nar(nar_model, nar_tokenizer, dataset, device, args.use_nar_gt)
        else:
            nar_decoder_output_ids = {
                dataset['id'][i]: {
                    f"encodec_{l}": dataset[f'encodec_{l}'][i]
                    for l in range(1, 8)
                }
                for i in range(len(dataset))
            }

        # output file
        write_decoder_output_ids(ar_decoder_output_ids, nar_decoder_output_ids, args.result_json)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--cascade_ar_nar",
        action="store_true"
    )
    parser.add_argument(
        "--use_ar_model",
        action="store_true"
    )
    parser.add_argument(
        "--use_nar_model",
        action="store_true"
    )
    parser.add_argument(
        "--ar_model",
        type=str,
        default="BartForConditionalGeneration"
    )
    parser.add_argument(
        "--nar_model",
        type=str,
        default="NARBartForConditionalGeneration"
    )
    parser.add_argument(
        "--ar_model_ckpt",
        type=str,
        default="./training_output/checkpoint-60000"
    )
    parser.add_argument(
        "--nar_model_ckpt",
        type=str,
        default="./training_output/checkpoint-60000"
    )
    parser.add_argument(
        "--use_nar_gt",
        action="store_true"
    )
    parser.add_argument(
        "--result_json",
        type=str,
        default="./result.json"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    
    args = parser.parse_args()
    
    return args

        
if __name__ == '__main__':
    args = parse_args()
    main(args)
