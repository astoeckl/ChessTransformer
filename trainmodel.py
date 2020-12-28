from transformers import AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


def run_training(model_type, train_path, test_path):

    tokenizer = AutoTokenizer.from_pretrained(model_type, eos_token='<|endoftext|>')
    train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)
    model = AutoModelWithLMHead.from_pretrained(model_type)

    training_args = TrainingArguments(
        output_dir="./data/models"+model_type,
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=1,  # number of training epochs
        per_device_train_batch_size=32,  # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        eval_steps=1000,  # Number of update steps between two evaluations.
        save_steps=500,  # after # steps model is saved
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.save_model()
