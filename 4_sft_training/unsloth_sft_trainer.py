"""
SFT Trainer for Decoder-Only Models with Multi-Turn Conversations

Trains a student model to learn from multi-turn conversations using supervised fine-tuning.
- Uses MultiTurnSFTDataset which handles chat templates and proper token masking
- Masks user/system tokens, only learns from assistant responses
- Built on HuggingFace Transformers Trainer for flexibility
"""

import unsloth
from unsloth import FastLanguageModel
import os
import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# Import the MultiTurnSFTDataset
from dataset import MultiTurnSFTDataset


# ---------- Data Collator ----------
@dataclass
class DataCollatorForMultiTurnSFT:
    """
    Custom collator for MultiTurnSFTDataset that handles variable-length sequences.
    Pads sequences to the longest in the batch for efficient training.
    """
    pad_token_id: int
    padding_side: str = "right"
    pad_to_multiple_of: int = 8
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract fields
        input_ids_list = [f['input_ids'] for f in features]
        labels_list = [f['labels'] for f in features]
        attention_mask_list = [f['attention_mask'] for f in features]
        
        # Pad sequences to max length in batch
        # Notice that training's padding_side is always 'right'
        input_ids = pad_sequence(input_ids_list,
                                 batch_first=True, 
                                 padding_value=self.pad_token_id)
        labels = pad_sequence(labels_list, 
                              batch_first=True, 
                              padding_value=-100)
        attention_mask = pad_sequence(attention_mask_list, 
                                     batch_first=True, 
                                     padding_value=0)
        
        if self.pad_to_multiple_of is not None:
            if self.padding_side == "right":
                # Pad to multiple of specified value
                seq_len = input_ids.size(1)
                if seq_len % self.pad_to_multiple_of != 0:
                    pad_len = self.pad_to_multiple_of - (seq_len % self.pad_to_multiple_of)
                    
                    input_ids = torch.nn.functional.pad(
                        input_ids, (0, pad_len), value=self.pad_token_id
                    )
                    labels = torch.nn.functional.pad(
                        labels, (0, pad_len), value=-100
                    )
                    attention_mask = torch.nn.functional.pad(
                        attention_mask, (0, pad_len), value=0
                    )
            else:
                raise NotImplementedError("Left padding not implemented in this collator.")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ---------- Visualization Helper ----------
def visualize_masking(
    dataset: MultiTurnSFTDataset,
    tokenizer: AutoTokenizer,
    num_examples: int = 1,
):
    """
    Visualize the masking behavior on training examples.
    Shows what tokens the model will learn from (labels != -100).
    """
    print("\n" + "="*80)
    print(f"MASKING VISUALIZATION - First {num_examples} Training Example(s)")
    print("="*80)
    
    for idx in range(min(num_examples, len(dataset))):
        print(f"\n{'='*80}")
        print(f"Example {idx}")
        print('='*80)
        
        # Get example
        batch = dataset[idx]
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        seq_len = len(input_ids)
        masked_tokens = (labels == -100).sum().item()
        unmasked_tokens = (labels != -100).sum().item()
        
        print(f"\n[1] Sequence Info:")
        print(f"  - Total length: {seq_len}")
        print(f"  - Masked tokens (user/system): {masked_tokens} ({masked_tokens/seq_len*100:.1f}%)")
        print(f"  - Learning tokens (assistant): {unmasked_tokens} ({unmasked_tokens/seq_len*100:.1f}%)")
        
        # Decode full conversation
        print(f"\n[2] Full Conversation (first 800 chars):")
        print("-" * 80)
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(full_text[:800] + "..." if len(full_text) > 800 else full_text)
        print("-" * 80)
        
        # Decode only what model learns (assistant responses)
        print(f"\n[3] Assistant Responses Only (what model learns, first 800 chars):")
        print("-" * 80)
        assistant_ids = input_ids[labels != -100]
        if len(assistant_ids) > 0:
            assistant_text = tokenizer.decode(assistant_ids, skip_special_tokens=True)
            print(assistant_text[:800] + "..." if len(assistant_text) > 800 else assistant_text)
        else:
            print("(No assistant tokens to learn from)")
        print("-" * 80)
    
    print("\n" + "="*80)
    print("END OF MASKING VISUALIZATION")
    print("="*80 + "\n")


# ---------- Data Loading ----------
def load_data(path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL or Parquet file.
    
    Expected format for MultiTurnSFTDataset:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }
    
    Args:
        path: Path to data file (.jsonl or .parquet)
    
    Returns:
        List of dictionaries with training examples
    
    [
{"role": "user", "content": ".<query> </query> <information>...</information> "},
{"role": "assistant", "content": "<reason>...</reason><summary>...</summary>"},
{"role": "tool", "content": "<information>...</information>"},
{"role": "assistant", "content": "<reason>...</reason><satisfy>yes</satisfy>"}
]
    
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Determine file type by extension
    suffix = file_path.suffix.lower()
    
    if suffix == '.jsonl':
        print(f"[Data] Loading JSONL file: {path}")
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}")
        return data
    
    elif suffix == '.parquet':
        print(f"[Data] Loading Parquet file: {path}")
        df = pd.read_parquet(path)
        # Convert DataFrame to list of dicts
        data = df.to_dict('records')
        return data
    
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Please use .jsonl or .parquet files."
        )


# ---------- Argument Parser ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT training for decoder-only models with multi-turn conversations"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training data (JSONL or Parquet format with 'messages' field)",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to evaluation data (optional, JSONL or Parquet format)",
    )
    parser.add_argument(
        "--messages_key",
        type=str,
        default="messages",
        help="Key in data dict containing messages list (default: 'messages')",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of training samples to use (-1 for all)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Pretrained model name or path (e.g., 'Qwen/Qwen2.5-7B-Instruct')",
    )
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./sft_output")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb", help="Logging platform (e.g., 'wandb', 'tensorboard', 'none')")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy: 'steps' or 'epoch'")
    parser.add_argument("--wandb_project", type=str, default="sft-training", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (default: auto-generated)")
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        help="Use Flash Attention 2 for efficient long-context training (requires flash-attn package)",
    )
    parser.add_argument(
        "--visualize_masking",
        action="store_true",
        help="Visualize token masking on first example before training",
    )
    
    # FSDP arguments
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Enable Fully Sharded Data Parallel (FSDP) for distributed training",
    )
    parser.add_argument(
        "--fsdp_config",
        type=str,
        default=None,
        help="Path to FSDP config JSON file (optional)",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=8,
        help="Pad sequences to a multiple of this value (optional)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision to save memory",
    )
    return parser.parse_args()


# ---------- Main Training Function ----------
def main():
    args = parse_args()
    
    # Load tokenizer and model 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name_or_path,
        max_seq_length = args.max_seq_length,
        # load_in_8bit = args.load_in_8bit,
        full_finetuning = True,  # Use LoRA adapters with quantized models
        dtype = None
    )
    FastLanguageModel.for_training(model)
    
    # Load training data using MultiTurnSFTDataset
    print(f"[Data] Loading training data from {args.train_file}")
    train_dataset = MultiTurnSFTDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        messages_key=args.messages_key,
        pad_mode="no_padding",  # Use data collator for dynamic padding
        max_samples=args.max_samples,
    )
    
    # Load evaluation data if provided
    eval_dataset = None
    if args.eval_file:
        print(f"[Data] Loading evaluation data from {args.eval_file}")
        eval_dataset = MultiTurnSFTDataset(
            data_path=args.eval_file,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            messages_key=args.messages_key,
            pad_mode="no_padding",
            truncation="error",
            max_samples=-1,  # Use all eval data
        )
    
    # Visualize masking if requested
    if args.visualize_masking:
        print("\n[Visualization] Checking masking behavior on first training example...")
        visualize_masking(
            dataset=train_dataset,
            tokenizer=tokenizer,
            num_examples=1,
        )
    
    # Create data collator for dynamic padding
    print("[Collator] Creating data collator for multi-turn conversations...")
    collator = DataCollatorForMultiTurnSFT(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )
    
    # Setup wandb if using it
    if args.report_to == "wandb":
        import wandb
        
        # Auto-generate run name if not provided
        if args.wandb_run_name is None:
            model_basename = os.path.basename(args.model_name_or_path)
            args.wandb_run_name = f"{model_basename}_bs{args.per_device_train_batch_size}x{args.gradient_accumulation_steps}_lr{args.learning_rate}"
        
        print(f"[Wandb] Project: {args.wandb_project}")
        print(f"[Wandb] Run name: {args.wandb_run_name}")
        
        # Initialize wandb
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
    
    # Setup training arguments
    # When eval_dataset exists, both save and eval must use "steps"
    save_strategy = "steps" if eval_dataset else "epoch"
    eval_strategy = "steps" if eval_dataset else "no"
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if eval_dataset else None,
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=args.save_total_limit,
        save_strategy=save_strategy,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,  # Important: keep all columns for custom dataset
        eval_strategy=eval_strategy,
        load_best_model_at_end=True if eval_dataset else False,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(args.output_dir, "final")
    print(f"\n[Saving] Saving final model to {final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    main()
