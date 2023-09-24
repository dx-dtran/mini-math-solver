import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader


def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan
    return answer


def compute_equation_acc(preds, labels):
    preds = [eval_equation(pred) for pred in preds]
    labels = [eval_equation(label) for label in labels]
    return np.mean(np.array(preds) == np.array(labels))


def tokenize_function(examples):
    # Assuming that your data has 'pred', 'expl', 'label', and 'rationale' as the keys
    pred_encodings = tokenizer(
        examples['pred'], truncation=True, padding='max_length', max_length=512, return_tensors='pt'
    )
    expl_encodings = tokenizer(
        examples['expl'], truncation=True, padding='max_length', max_length=512, return_tensors='pt'
    )
    label_encodings = tokenizer(
        examples['label'], truncation=True, padding='max_length', max_length=512, return_tensors='pt'
    )
    rationale_encodings = tokenizer(
        examples['rationale'], truncation=True, padding='max_length', max_length=512, return_tensors='pt'
    )

    return {
        'pred': pred_encodings,
        'expl': expl_encodings,
        'label': label_encodings,
        'rationale': rationale_encodings
    }


def collate_fn(batch):
    def stack_encoding(key):
        return {
            'input_ids': torch.stack([torch.tensor(item[key]['input_ids']).squeeze() for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item[key]['attention_mask']).squeeze() for item in batch]),
        }

    pred = stack_encoding('pred')
    expl = stack_encoding('expl')
    label_input_ids = torch.stack([torch.tensor(item['label']['input_ids']).squeeze() for item in batch])
    rationale_input_ids = torch.stack([torch.tensor(item['rationale']['input_ids']).squeeze() for item in batch])

    return {
        'pred': pred,
        'expl': expl,
        'label': {'input_ids': label_input_ids},
        'rationale': {'input_ids': rationale_input_ids},
    }


# Checking for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')

tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')
student_model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base')
student_model = student_model.to(device)

# Load the dataset
dataset = load_dataset('json', data_files='svamp_train.json', split='train')

# Apply the tokenize function to all examples in the dataset
tokenized_dataset = dataset.map(tokenize_function)

# Create a DataLoader to handle batching
train_loader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Load the dataset
dataset = load_dataset('json', data_files='svamp_train.json', split='train')

tokenized_dataset = dataset.map(tokenize_function)

optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
num_epochs = 10
alpha = 0.5  # Assume an equal weight for simplicity, adjust as needed
accumulation_steps = 16

# Training loop
for epoch in range(num_epochs):
    student_model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Move batch data to the same device as the model (i.e., GPU if available)
        pred_inputs = batch['pred']['input_ids'].to(device)
        pred_attention_mask = batch['pred']['attention_mask'].to(device)
        pred_labels = batch['label']['input_ids'].to(device)

        expl_inputs = batch['expl']['input_ids'].to(device)
        expl_attention_mask = batch['expl']['attention_mask'].to(device)
        expl_labels = batch['rationale']['input_ids'].to(device)

        # Forward pass for prediction
        pred_outputs = student_model(input_ids=pred_inputs, attention_mask=pred_attention_mask, labels=pred_labels)

        # Forward pass for explanation
        expl_outputs = student_model(input_ids=expl_inputs, attention_mask=expl_attention_mask, labels=expl_labels)

        # Combined loss
        loss = alpha * pred_outputs.loss + (1 - alpha) * expl_outputs.loss
        loss = loss / accumulation_steps  # Normalize the loss

        # Backward pass and optimization
        loss.backward()
        running_loss += loss.item() * accumulation_steps  # undo the division to accumulate the actual loss
        if (i + 1) % accumulation_steps == 0:  # Step is zero-indexed
            optimizer.step()  # Update the model parameters
            optimizer.zero_grad()

    print(f'Training loss epoch {epoch + 1}: {running_loss / len(train_loader)}')
