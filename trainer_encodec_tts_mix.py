import argparse

from datasets import load_dataset
from jiwer import wer
from transformers import (AutoTokenizer, BartForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

from mix_bart import MIXBartForConditionalGeneration

TRAIN_ARGS = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,
    predict_with_generate=True,
    fp16=True,
    gradient_accumulation_steps=2,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="voidful/librispeech_encodec")
    parser.add_argument("--train_splits", type=str, nargs="+", default=["trainclean100"])
    parser.add_argument("--eval_splits", type=str, nargs="+", default=["validationclean"])
    parser.add_argument("--model", type=str, default="MIXBartForConditionalGeneration")
    parser.add_argument("--model_card", type=str, default="voidful/bart-base-unit")
    parser.add_argument("--ar_layers", type=int, nargs="+", default=[0])
    parser.add_argument("--nar_layers", type=int, nargs="+", default=list(range(1, 8)))
    
    args = parser.parse_args()
    return args


def pad_sequences(sequences, max_length, padding_value):
    return [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]


def filter_examples(example):
    return len(example[f"encodec_0"]) <= 1000


def process_data_to_model_inputs(batch, args, tokenizer):
    ar_layers = args.ar_layers
    assert ar_layers in [[0], []], "Wrong ar_layers, should be [] or [0]"
    nar_layers = args.nar_layers
    assert 0 not in nar_layers, "Wrong nar_layers, should not contain 0"
    
    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    labels = []
    non_autoregressive_mode = []
    
    max_length = 1023
    
    for b in range(len(batch['text'])):
        data = tokenizer(batch['text'][b], padding='max_length',
                         truncation=True, max_length=max_length)
        # AR data
        for l in ar_layers:
            encode_input = tokenizer.convert_tokens_to_ids([
                f'v_tok_{u}' for u in batch[f'encodec_{l}'][b]])
            decoder_input_id = [tokenizer.bos_token_id] + encode_input
            label = encode_input + [tokenizer.eos_token_id]
            input_ids.append(data['input_ids'])
            attention_mask.append(data['attention_mask'])
            decoder_input_ids.append(decoder_input_id)
            labels.append(label)
            non_autoregressive_mode.append(False)
        
        # NAR data
        for l in nar_layers:
            decoder_input_id = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (l - 1) * 1024}" for u in batch[f'encodec_{l - 1}'][b]])
            label = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + l * 1024}" for u in batch[f'encodec_{l}'][b]])
            input_ids.append(data['input_ids'])
            attention_mask.append(data['attention_mask'])
            decoder_input_ids.append(decoder_input_id)
            labels.append(label)
            non_autoregressive_mode.append(True)

    decoder_input_ids = pad_sequences(decoder_input_ids, max_length=max_length,
                                      padding_value=tokenizer.pad_token_id)
    labels = pad_sequences(labels, max_length=max_length,
                           padding_value=tokenizer.pad_token_id)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "non_autoregressive_mode": non_autoregressive_mode
    }


def get_dataset(tokenizer, args):
    train_dataset = load_dataset(args.dataset, "train", split='+'.join(args.train_splits))
    eval_dataset = load_dataset(args.dataset, "eval", split='+'.join(args.eval_splits))
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


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
    wer_value = wer(decoded_labels, decoded_preds)
    print("pred_result")
    print("=================================")
    for i in range(10):
        print(decoded_labels[i], " ///// ", decoded_preds[i])
    print("=================================")
    return {"wer": wer_value}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_card)
    model = eval(args.model).from_pretrained(args.model_card)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model)
    train_dataset, eval_dataset = get_dataset(tokenizer, args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=TRAIN_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda preds : compute_metrics(preds, tokenizer)
    )

    trainer.train()


if __name__ == '__main__':
    args = get_args()
    main(args)