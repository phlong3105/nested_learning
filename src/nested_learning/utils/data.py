"""Data loading utilities."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


def get_dataloader(
    dataset_name='wikitext',
    dataset_config='wikitext-103-raw-v1',
    split='train',
    tokenizer_name='gpt2',
    batch_size=8,
    max_length=1024,
    num_workers=4,
):
    """
    Create a dataloader for language modeling.

    Args:
        dataset_name: Name of the dataset
        dataset_config: Configuration of the dataset
        split: Dataset split ('train', 'validation', 'test')
        tokenizer_name: Name of the tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading

    Returns:
        DataLoader
    """
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize function
    def tokenize_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        # For language modeling, labels are the same as input_ids
        result['labels'] = result['input_ids'].clone()
        return result

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Create dataloader
    tokenized_dataset.set_format(type='torch')
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
    )

    return dataloader