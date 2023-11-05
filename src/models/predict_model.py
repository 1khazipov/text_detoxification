from transformers import AutoTokenizer


def predict(trainer, dataset, model_checkpoint='t5-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer(dataset['train'])
    predictions, labels, metrics = trainer.predict(dataset["test"], metric_key_prefix="predict")
    my_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return my_predictions, labels, metrics
