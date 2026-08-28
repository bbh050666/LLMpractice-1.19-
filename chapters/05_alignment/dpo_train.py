"""使用 chosen/rejected 偏好对进行 LoRA DPO 训练。"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct Preference Optimization")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/dpo")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=1,
        max_length=512,
        max_prompt_length=256,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = DPOTrainer(
        model=args.model_id,
        ref_model=None,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
