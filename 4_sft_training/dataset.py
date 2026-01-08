"""
Multi-Turn SFT Dataset for training with conversation data.
Handles proper token masking so the model only learns from assistant responses.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset



class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn supervised fine-tuning.
    Masks user/system tokens and only learns from assistant responses.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 32768,
        messages_key: str = "messages",
        pad_mode: str = "no_padding",
        truncation: str = "auto",
        max_samples: int = -1,
    ):
        """
        Args:
            data_path: Path to JSONL or Parquet file
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            messages_key: Key containing messages list
            pad_mode: "no_padding" for dynamic padding with collator
            truncation: "auto" or "error"
            max_samples: Max samples to load (-1 for all)
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.messages_key = messages_key
        self.pad_mode = pad_mode
        self.truncation = truncation
        
        # Load data
        self.data = self._load_data()
        
        # Limit samples if specified
        if max_samples > 0:
            self.data = self.data[:max_samples]
        
        print(f"[Dataset] Loaded {len(self.data)} examples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL or Parquet file"""
        file_path = Path(self.data_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
        if file_path.suffix.lower() == '.jsonl':
            return self._load_jsonl(str(file_path))
        elif file_path.suffix.lower() == '.parquet':
            return self._load_parquet(str(file_path))
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
        return data
    
    def _load_parquet(self, path: str) -> List[Dict]:
        """Load Parquet file"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for parquet support")
        
        df = pd.read_parquet(path)
        return df.to_dict('records')
    
    def _build_prompt_with_masking(self, messages: List[Dict[str, str]]) -> tuple:
        """
        Build prompt from messages and create labels with proper masking.
        Only assistant responses are unmasked (label != -100).
        
        Returns:
            (input_ids, labels, attention_mask)
        """
        input_ids = []
        labels = []
        """

--------messages--------- [{'role': 'user', 'content': 'How do I start learning Python programming?'}, {'role': 'assistant', 'content': 'Here are the steps to learn Python:\n1. Install Python from python.org\n2. Learn basic syntax: variables, data types, control flow\n3. Study functions and modules\n4. Practice with real projects\n5. Learn popular libraries like NumPy, Pandas, and Matplotlib\nI recommend starting with official Python tutorials and online courses on platforms like Coursera or Codecademy.'}]
        --------text--------- <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
        <|im_start|>user
        How do I start learning Python programming?<|im_end|>
        <|im_start|>assistant
        Here are the steps to learn Python:
        1. Install Python from python.org
        2. Learn basic syntax: variables, data types, control flow
        3. Study functions and modules
        4. Practice with real projects
        5. Learn popular libraries like NumPy, Pandas, and Matplotlib
        I recommend starting with official Python tutorials and online courses on platforms like Coursera or Codecademy.<|im_end|>
        """
        # Use chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use the tokenizer's built-in chat template
            try:
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fallback to manual building
                text = self._build_prompt_manual(messages)
        else:
            # Manual prompt building
            text = self._build_prompt_manual(messages)
        
        # Tokenize full conversation
        # print(text)
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=(self.truncation == "auto"),
            return_tensors=None,  # Return lists, not tensors
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings.get('attention_mask', [1] * len(input_ids))
        
        # Create labels with masking
        # Only learn from assistant content, mask everything else
        labels = [-100] * len(input_ids)  # Start with all masked
        
        # Process each message to unmask only assistant content
        for i, message in enumerate(messages):
            # print("--------messages---------", messages)
            role = message.get('role', 'user')
            # print("--------role---------", role)
            content = message.get('content', '')
            # print("--------content---------", content)
            
            if role != 'assistant':
                # Skip non-assistant messages
                continue
            
            # For assistant messages, find exact token positions using text-based approach
            # Get text up to current message (including it)
            text_up_to_current = self.tokenizer.apply_chat_template(
                messages[:i+1],
                tokenize=False,
                add_generation_prompt=False,
            )
            # print("--------text_up_to_current---------", text_up_to_current)
            # Get text before current message for boundary check
            if i == 0:
                text_before = ""
                text_before_len = 0
            else:
                text_before = self.tokenizer.apply_chat_template(
                    messages[:i],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                text_before_len = len(text_before)
            
            # Find content in the CURRENT message range (text_up_to_current)
            # Use rfind to find the LAST occurrence (should be in current message)
            content_pos_in_current = text_up_to_current.rfind(content)
            
            if content_pos_in_current != -1:
                # Verify structure: text_up_to_current should start with text_before
                # (or be equal if i==0)
                if (i == 0) or (text_up_to_current.startswith(text_before) and content_pos_in_current >= text_before_len):
                    # Content position is verified to be in current message
                    # Found content, tokenize text before content to get start position
                    text_before_content = text_up_to_current[:content_pos_in_current]
                    tokens_before = self.tokenizer(
                        text_before_content,
                        return_tensors=None,
                    )['input_ids']
                    content_start_token = len(tokens_before)
                    
                    # Tokenize text up to end of content to get end position
                    text_up_to_content_end = text_up_to_current[:content_pos_in_current + len(content)]
                    tokens_up_to_end = self.tokenizer(
                        text_up_to_content_end,
                        return_tensors=None,
                    )['input_ids']
                    content_end_token = len(tokens_up_to_end)
                    
                    # Unmask assistant content tokens
                    for idx in range(content_start_token, min(content_end_token, len(labels))):
                        labels[idx] = input_ids[idx]
        
        # Handle truncation
        if len(input_ids) > self.max_length:
            # if self.truncation == "error":
            #     raise ValueError(
            #         f"Sequence length {len(input_ids)} exceeds max_length {self.max_length}"
            #     )
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
            )
        else:
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
            )
    
    def _build_prompt_manual(self, messages: List[Dict[str, str]]) -> str:
        """Manually build prompt from messages"""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user').lower()
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")
        
        return "\n".join(prompt_parts)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example"""
        example = self.data[idx]
        
        # Extract messages
        # print("------self.messages_key------", self.messages_key)
        messages = example.get(self.messages_key, [])
        # print("------messages------", messages)
        if not messages:
            raise ValueError(f"No messages found in example {idx}")
        
        # Build prompt and get labels
        input_ids, labels, attention_mask = self._build_prompt_with_masking(messages)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
