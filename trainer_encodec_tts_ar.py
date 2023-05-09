from argparse import ArgumentParser, Namespace

from datasets import load_dataset
from jiwer import wer
from transformers import (AutoTokenizer, BartForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

import wandb

wandb.init(project="encodec_tts", 
           name="bart-base-ar",
)


TRAIN_ARGS = Seq2SeqTrainingArguments(
    output_dir="./training_output/ar",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.08,
    weight_decay=1e-4,
    logging_dir="./logs/ar",
    logging_steps=500,
    save_steps=5000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=5000,
    predict_with_generate=True,
    fp16=True,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    generation_max_length=1024,
    report_to="wandb",
)


def pad_sequences(sequences, max_length, padding_value):
    return [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]


def filter_examples(example):
    return len(example[f"encodec_0"]) <= 1000


def process_data_to_model_inputs(batch, tokenizer):
    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    labels = []

    max_length = 1023

    for b in range(len(batch['text'])):
        data = tokenizer(batch["text"][b], padding='max_length', truncation=True, max_length=max_length)
        encode_input = tokenizer.convert_tokens_to_ids([f"v_tok_{u}" for u in batch[f'encodec_{0}'][b]])
        decoder_input_id = [tokenizer.bos_token_id] + encode_input
        label = encode_input + [tokenizer.eos_token_id]
        
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])
        decoder_input_ids.append(decoder_input_id)
        labels.append(label)

    # Pad decoder_input_ids and labels
    decoder_input_ids = pad_sequences(decoder_input_ids, max_length=max_length, padding_value=tokenizer.pad_token_id)
    labels = pad_sequences(labels, max_length=max_length, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
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
        fn_kwargs={"tokenizer": tokenizer}
    )
    eval_dataset = eval_dataset.map(
        process_data_to_model_inputs,
        remove_columns=eval_dataset.column_names,
        batched=True,
        batch_size=TRAIN_ARGS.per_device_eval_batch_size,
        fn_kwargs={"tokenizer": tokenizer}
    )

    return train_dataset, eval_dataset


def main(args):
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    train_dataset, eval_dataset = get_dataset(tokenizer, args)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=TRAIN_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda preds : compute_metrics(preds, tokenizer),
    )

    trainer.train()

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="voidful/librispeech_encodec")
    parser.add_argument("-t", "--train_splits", type=str, nargs="+",
                        default=["trainclean100", "trainclean360", "trainother500"])
    parser.add_argument("-e", "--eval_splits", type=str, nargs="+",
                        default=["validationclean"])
    parser.add_argument("-m", "--model_name", type=str, default="voidful/bart-base-unit")

    args = parser.parse_args()    
    return args
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
