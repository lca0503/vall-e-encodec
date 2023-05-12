from argparse import ArgumentParser, Namespace

import torch
import wandb
from datasets import load_dataset
from jiwer import wer
from transformers import (AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

from encodec_model.nar_bart_model import NARBartForConditionalGeneration

wandb.init(project="encodec_tts", 
           name="bart-base-nar-full",
)


TRAIN_ARGS = Seq2SeqTrainingArguments(
    output_dir="./training_output/nar_full",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.08,
    weight_decay=1e-4,
    logging_dir="./logs/nar_full",
    logging_steps=500,
    save_steps=10000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=10000,
    predict_with_generate=False,
    fp16=True,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    report_to="wandb",
)


def pad_sequences(sequences, max_length, padding_value):
    return [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]


def get_attention_mask(seq_len, max_length):
    return [1] * seq_len + [0] * (max_length - seq_len)


def filter_examples(example):
    return len(example[f"src_encodec_0"]) <= 700


def get_encodec_units(data, b, split="src"):
    encodec_units = [
        [f"v_tok_{u + l * 1024}" for u in data[f"{split}_encodec_{l}"][b]]
        for l in range(8)
    ]
    for e in encodec_units:
        assert len(e) == len(encodec_units[0]), f"Inconsistent encodec unit length"
    return encodec_units


def process_data_to_model_inputs(batch, args, tokenizer):
    nar_layers = args.nar_layers

    input_ids = []
    encodec_unit_ids = []
    attention_masks = []
    decoder_input_ids = []
    labels = []

    max_length = 1023
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    for b in range(len(batch["instruction"])):
        # encoder input
        instruction_ids = tokenizer(batch["instruction"][b])["input_ids"][1 : -1]
        transcription_ids = tokenizer(batch["transcription"][b])["input_ids"][1 : -1]
        encodec_layer0 = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u}" for u in batch[f"src_encodec_{0}"][b]])
        encoder_input = [bos_token_id] + \
                        instruction_ids + [sep_token_id] + \
                        transcription_ids + [sep_token_id] + \
                        encodec_layer0 + [eos_token_id]
        seq_len = len(encoder_input)
        attention_mask = get_attention_mask(seq_len, max_length)

        for l in nar_layers:
            # decoder input
            decoder_input_id = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (l - 1) * 1024}" for u in batch[f"tgt_encodec_{l - 1}"][b]])
            label = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + l * 1024}" for u in batch[f"tgt_encodec_{l}"][b]])
            
            input_ids.append(encoder_input)
            attention_masks.append(attention_mask)
            decoder_input_ids.append(decoder_input_id)
            labels.append(label)

    input_ids = pad_sequences(input_ids, max_length=max_length,
                              padding_value=pad_token_id)
    decoder_input_ids = pad_sequences(decoder_input_ids, max_length=max_length,
                                      padding_value=pad_token_id)
    labels = pad_sequences(labels, max_length=max_length,
                           padding_value=pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }


def get_dataset(tokenizer, args):
    train_dataset = load_dataset(args.dataset, "train", split="+".join(args.train_splits))
    eval_dataset = load_dataset(args.dataset, "eval", split="+".join(args.eval_splits))
    
    train_dataset = train_dataset.filter(filter_examples)
    eval_dataset = eval_dataset.filter(filter_examples)

    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=TRAIN_ARGS.per_device_train_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer}
    )
    eval_dataset = eval_dataset.map(
        process_data_to_model_inputs,
        remove_columns=eval_dataset.column_names,
        batched=True,
        batch_size=TRAIN_ARGS.per_device_eval_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer}
    )

    return train_dataset, eval_dataset


def preprocess_logits_for_metrics(ret, labels):
    logits, encoder_last_hidden_state = ret
    predictions = torch.argmax(logits, axis=-1)

    return predictions


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    predictions = [prediction[:len(label)] for prediction, label in zip(predictions, labels)]
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer_value = wer([" ".join(filter(None, i.split("v_tok_"))) for i in decoded_labels],
                    [" ".join(filter(None, i.split("v_tok_"))) for i in decoded_preds])
    
    print("pred_result")
    print("=================================")
    for i in range(10):
        print("target:", labels[i])
        print("pred:", predictions[i])
        print("-----------------")
    print("=================================")
    
    return {"wer": wer_value}


def main(args):
    model = NARBartForConditionalGeneration.from_pretrained(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
    
    train_dataset, eval_dataset = get_dataset(tokenizer, args)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=TRAIN_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda preds : compute_metrics(preds, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="lca0503/soxdata_small_encodec")
    parser.add_argument("-t", "--train_splits", type=str, nargs="+", default=["train"])
    parser.add_argument("-e", "--eval_splits", type=str, nargs="+", default=["validation"])
    parser.add_argument("-m", "--model_name", type=str, default="voidful/bart-base-unit")
    parser.add_argument("-n", "--nar_layers", type=int, nargs="+", default=list(range(1, 8)))

    args = parser.parse_args()    
    return args

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
