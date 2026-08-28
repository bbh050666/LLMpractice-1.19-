"""用 TRL SFTTrainer 和 PEFT LoRA 进行监督微调。"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 监督微调")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/sft-lora")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    def format_chat(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=max(1, args.max_steps),
        max_seq_length=512,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=args.model_id,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=format_chat,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
