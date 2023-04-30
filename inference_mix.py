import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BartForConditionalGeneration
)
from nar_bart import NARBartForConditionalGeneration
import torch
import json


def get_args():
    parser = argparse.ArgumentParser()
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
        default="./training_output/checkpoint-300000"
    )
    parser.add_argument(
        "--nar_model_ckpt",
        type=str,
        default="voidful/bart-base-unit"
    )
    parser.add_argument(
        "--result_json",
        type=str,
        default="./result.json"
    )
    
    args = parser.parse_args()
    return args


def inference_ar(ar_model, ar_tokenizer, dataset):
    decoder_outputs = {}
    for i in range(len(dataset)):
        file_id = dataset['id'][i]
        inputs = ar_tokenizer([dataset['text'][i]], padding='max_length', 
                                 truncation=True, max_length=1024, return_tensors="pt")
        output_ids = ar_model.generate(input_ids=inputs['input_ids'], max_length=1024)
        decode_output = ar_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        decoder_outputs[file_id] = [token.split('v_tok_')[0] for token in decode_output.split(' ')]
    return decoder_outputs


def filter_token(output, target_tokens):
    for i in range(output.logits.size(-1)):
        if i not in target_tokens:
            output[:, :, i] = float('-inf')
    return output


def get_target_token_sets(tokenizer):
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


def inference_nar(nar_model, nar_tokenizer, dataset):
    decoder_outputs = {}
    target_token_sets = get_target_token_sets(nar_tokenizer)
    
    for i in range(len(dataset)):
        file_id = dataset['id'][i]
        inputs = nar_tokenizer([dataset['text'][i]], padding='max_length', 
                                 truncation=True, max_length=1024, return_tensors="pt")
        decoder_outputs[file_id] = decoder_outputs.get(file_id, {})
        
        for l in range(7):
            decoder_input_id = nar_tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + l * 1024}" for u in dataset[f'encodec_{l}'][i]])
            output = nar_model(inputs['input_id'], decoder_input_ids=[decoder_input_id])
            output = filter_token(output, target_token_sets[l])
            decode_ids = torch.argmax(output.logits, dim=-1)
            decode_output = nar_tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
            
            decoder_outputs[file_id][f'encodec_{l + 1}'] = \
                [token.split('v_tok_')[0] for token in decode_output.split(' ')]
    
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
    # dataset
    dataset = load_dataset("voidful/librispeech_encodec", split="trainclean100")
    dataset = dataset.filter(lambda x : len(x[f"encodec_0"]) <= 1000)
    dataset = dataset.shuffle(seed=42).select(range(30))
    
    if args.use_ar_model:
        ar_model = eval(args.ar_model).from_pretrained(args.ar_model_ckpt)
        ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_model_ckpt)
        ar_decoder_output_ids = inference_ar(ar_model, ar_tokenizer, dataset)
    else:
        ar_decoder_output_ids = {
            dataset['id'][i]: dataset['encodec_0'][i]
            for i in range(len(dataset))
        }
    if args.use_nar_model:
        nar_model = eval(args.nar_model).from_pretrained(args.nar_model_ckpt)
        nar_tokenizer = AutoTokenizer.from_pretrained(args.nar_model_ckpt)
        nar_decoder_output_ids = inference_nar(nar_model, nar_tokenizer, dataset)
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


if __name__ == '__main__':
    args = get_args()
    main(args)