import torch

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, outputs, tokenizer):
        self.prompts = prompts
        self.outputs = outputs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        input_ids = []
        labels = []
        self.tokenizer.truncation_side = "left"
        prompt_enc = self.tokenizer(self.prompts[idx], truncation=True, add_special_tokens=False, max_length=1024)["input_ids"]
        input_ids.extend(prompt_enc)
        labels.extend([-100] * len(prompt_enc))
        self.tokenizer.truncation_side = "right"
        output_enc = self.tokenizer(self.outputs[idx], truncation=True, add_special_tokens=False, max_length=8192)["input_ids"]
        input_ids.extend(output_enc)
        labels.extend(output_enc)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        }


def collate_fn(batch, tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        padding_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_len)
        attention_mask.append(item["attention_mask"] + [0] * padding_len)
        labels.append(item["labels"] + [-100] * padding_len)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }