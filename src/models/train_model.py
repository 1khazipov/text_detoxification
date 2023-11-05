from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from metrics import compute_metrics


def trainer(dataset, model_checkpoint = 't5-base'):  # 's-nlp/t5-paranmt-detox'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    batch_size = 32
    model_name = model_checkpoint.split("/")[-1]
    toxic_str = "toxic"
    nontoxic_str = "nontoxic"
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{toxic_str}-to-{nontoxic_str}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=20,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        generation_max_length=80,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer(dataset['train'])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    splitted_dataset = dataset["train"].select(range(20000)).train_test_split(test_size=0.1)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=splitted_dataset["train"],
        eval_dataset=splitted_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model()

    return trainer
