import random
from argparse import ArgumentParser, Namespace

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from encodec import EncodecModel
from encodec_model.nar_bart_model import NARBartForConditionalGeneration
from transformers import AutoTokenizer, BartForConditionalGeneration


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def nar_decode(model, tokenizer, inputs, batch_code, layer=0):
    base_input = inputs
    base_input['decoder_input_ids'] = batch_code
    decode_nar = model.forward(**base_input).logits

    id_range_start, id_range_end = tokenizer.convert_tokens_to_ids(
        f'v_tok_{0 + 1024 * layer}'), tokenizer.convert_tokens_to_ids(f'v_tok_{1024 + 1024 * layer}')

    # Create a tensor where values are equal to their own indices
    indices = torch.arange(decode_nar.size(-1)).to(decode_nar.device)

    # Create a mask for the range
    mask = (indices >= id_range_start) & (indices < id_range_end)

    # Set values out of range to very low value
    decode_nar_masked = torch.where(mask, decode_nar, torch.tensor(float('-inf')).to(decode_nar.device))

    # Get the argmax within the range
    return torch.argmax(decode_nar_masked, dim=-1)


def ground_truth_only(tokenizer, dataset, device):
    layer_list = []
    for layer_i in range(8):
        encode_input = tokenizer(
            "".join([f"v_tok_{u + layer_i * 1024}" for u in dataset[f'encodec_{layer_i}'][0]]),
            return_tensors='pt', add_special_tokens=False)
        encode_input = encode_input['input_ids'].to(device)
        layer_list.append(encode_input)

    return layer_list


def cascade_ar_nar(ar_model, nar_model, ar_tokenizer, nar_tokenizer, dataset, device):
    layer_list = []

    # Get AR prediction
    inputs = ar_tokenizer(dataset["text"][0], padding="max_length", truncation=True,
                          max_length=1024, return_tensors="pt")
    inputs = inputs.to(device)

    bad_words_ids = [[ar_tokenizer.convert_tokens_to_ids(f'v_tok_{i}')] for i in range(1025, 1024*8)]
    decode_ar = ar_model.generate(**inputs, max_length=1024, num_beams=1,
                                  do_sample=True, use_cache=True, bad_words_ids=bad_words_ids)

    layer_list.append(decode_ar[:, 1:-1])

    # Iterative predict NAR code
    for layer in range(1, 8):
        layer_list.append(nar_decode(nar_model, nar_tokenizer, layer_list[-1], layer))

    return layer_list

    
def nar_model_only(model, tokenizer, dataset, device):
    layer_list = []

    inputs = tokenizer(dataset["text"][0], padding="max_length", truncation=True,
                       max_length=1024, return_tensors="pt")
    inputs = inputs.to(device)

    # Use ground truth AR prediction
    encode_input = tokenizer(
        "".join([f"v_tok_{u}" for u in dataset[f'encodec_0'][0]]),
        return_tensors='pt', add_special_tokens=False)
    encode_input_ids = encode_input['input_ids'].to(device)
    layer_list.append(encode_input_ids)

    # Iterative predict NAR code
    for layer in range(1, 8):
        layer_list.append(nar_decode(model, tokenizer, inputs, layer_list[-1], layer))

    return layer_list


def convert_to_encode_code(tokenizer, layer_list):
    encodec_code = []
    for layer, layer_ids in enumerate(tokenizer.batch_decode(torch.cat(layer_list))):
        layer_ids = layer_ids.replace("</s>", "")
        encodec_code.append([int(i) - layer * 1024 for i in layer_ids.split('v_tok_') if len(i) > 0])

    return encodec_code

        
def synthesize_audio(encodec_code, device):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)

    encodec_input = torch.tensor(encodec_code).unsqueeze(0)
    encodec_input = encodec_input.to(device)
    audio = model.decode([(encodec_input, None)]).cpu().detach().numpy()[0]

    return audio
    

def main(args):
    set_seed(args.seed)
    device = args.device
    
    dataset = load_dataset(args.dataset, split="+".join(args.splits))
    dataset = dataset.filter(lambda x : len(x[f"encodec_0"]) <= 1000)
    dataset = dataset.shuffle(args.seed).select(range(1))
    
    if args.ground_truth_only:
        tokenizer = AutoTokenizer.from_pretrained(args.ground_truth_model_name)
        layer_list = ground_truth_only(tokenizer, dataset, device)
        encodec_code = convert_to_encode_code(tokenizer, layer_list)    
        audio = synthesize_audio(encodec_code, device)
        sf.write(args.ground_truth_output_path, np.ravel(audio), samplerate=24000)

    if args.cascade_ar_nar:
        ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_checkpoint)
        ar_model = BartForConditionalGeneration.from_pretrained(args.ar_checkpoint)
        ar_model.to(device)

        nar_tokenizer = AutoTokenizer.from_pretrained(args.nar_checkpoint)
        nar_model = NARBartForConditionalGeneration.from_pretrained(args.nar_checkpoint)
        nar_model.to(device)

        layer_list = cascade_ar_nar(ar_model, nar_model, ar_tokenizer, nar_tokenizer, dataset, device)
        encodec_code = convert_to_encode_code(nar_tokenizer, layer_list)    
        audio = synthesize_audio(encodec_code, device)
        sf.write(args.cascade_output_path, np.ravel(audio), samplerate=24000)
            
    if args.nar_model_only:
        nar_tokenizer = AutoTokenizer.from_pretrained(args.nar_checkpoint)
        nar_model = NARBartForConditionalGeneration.from_pretrained(args.nar_checkpoint)
        nar_model.to(device)

        layer_list = nar_model_only(nar_model, nar_tokenizer, dataset, device)
        encodec_code = convert_to_encode_code(nar_tokenizer, layer_list)    
        audio = synthesize_audio(encodec_code, device)
        sf.write(args.nar_output_path, np.ravel(audio), samplerate=24000)
        
        
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="voidful/librispeech_encodec")
    parser.add_argument("-s", "--splits", type=str, nargs="+", default=["trainclean100"])
    
    parser.add_argument("--ground_truth_only", action="store_true")
    parser.add_argument("--cascade_ar_nar", action="store_true")
    parser.add_argument("--nar_model_only", action="store_true")
    
    parser.add_argument("--ground_truth_model_name", type=str, default="voidful/bart-base-unit")
    parser.add_argument("--ar_checkpoint", type=str, default="/work/b08902123/SpeechChatGPT/previous_ckpt/ar/checkpoint-105000/")
    parser.add_argument("--nar_checkpoint", type=str, default="/work/b08902123/SpeechChatGPT/previous_ckpt/nar/checkpoint-105000/")

    parser.add_argument("--ground_truth_output_path", type=str, default="ground_truth.wav")
    parser.add_argument("--cascade_output_path", type=str, default="cascade.wav")
    parser.add_argument("--nar_output_path", type=str, default="nar.wav")
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=torch.device, default="cuda")
    
    args = parser.parse_args()    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
