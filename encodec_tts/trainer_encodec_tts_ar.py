from datasets import load_dataset
from jiwer import wer
from transformers import (AutoTokenizer, BartForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, GenerationConfig)

import wandb


# Load dataset and tokenizer
train_dataset = load_dataset("voidful/librispeech_encodec", split="trainclean100+trainclean360+trainother500")
valid_dataset = load_dataset("voidful/librispeech_encodec", split="validationclean")
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")

# Load model
model = BartForConditionalGeneration.from_pretrained("voidful/bart-base-unit")

    
# Set training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output_ar",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.08,
    weight_decay=1e-4,
    logging_dir="./logs_ar",
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
)

# Filter long examples
def filter_examples(example):
    return len(example[f"encodec_0"]) <= 1000

train_dataset = train_dataset.filter(filter_examples)
valid_dataset = valid_dataset.filter(filter_examples)

def pad_sequences(sequences, max_length, padding_value):
    return [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]

def process_data_to_model_inputs(batch):
    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    labels = []

    max_length = 1023  # You can set this to a suitable value based on your dataset

    for b in range(len(batch['text'])):
        # first layer AR data
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

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    remove_columns=train_dataset.column_names,
    batched=True,
    batch_size=training_args.per_device_train_batch_size
)
valid_dataset = valid_dataset.map(
    process_data_to_model_inputs,
    remove_columns=valid_dataset.column_names,
    batched=True,
    batch_size=training_args.per_device_eval_batch_size
)

# Define a data collator to handle tokenization
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
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

name = 'libri960_ar_wr0.08_lr1e-4'
wandb.init(name)

# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
trainer.evaluate()
