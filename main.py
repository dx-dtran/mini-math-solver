import torch
import numpy as np
import time
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
        examples['pred'], truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    expl_encodings = tokenizer(
        examples['expl'], truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    label_encodings = tokenizer(
        examples['label'], truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    rationale_encodings = tokenizer(
        examples['rationale'], truncation=True, padding='max_length', max_length=128, return_tensors='pt'
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

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
student_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
student_model = student_model.to(device)

dataset = load_dataset('json', data_files='svamp_train.json', split='train')

# Split the dataset
train_val_dataset = dataset.train_test_split(test_size=0.1)

# Apply the tokenize function to all examples in the dataset
tokenized_train_dataset = train_val_dataset["train"].map(tokenize_function)
tokenized_val_dataset = train_val_dataset["test"].map(tokenize_function)

BATCH_SIZE = 8

# Create DataLoaders to handle batching
train_loader = DataLoader(tokenized_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(tokenized_val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

num_epochs = 100
accumulation_steps = 8
total_data_points = len(train_loader)
effective_batch_size = BATCH_SIZE * accumulation_steps  # The effective batch size
total_iterations = (total_data_points // effective_batch_size) * num_epochs  # Recalculate total_iterations

print("num iterations: ", total_iterations)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)
alpha = 0.5  # Assume an equal weight for simplicity, adjust as needed
start = time.time()

# TODO: calculate the LLM's math performance before we do training
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
            scheduler.step()

    print(f'Training loss epoch {epoch + 1}: {running_loss / len(train_loader)}')
    print('time: {:0.2f} seconds'.format(time.time() - start))

    # Evaluation
    student_model.eval()
    val_preds, val_labels = [], []
    inputs_list, expl_list = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            pred_inputs = batch['pred']['input_ids'].to(device)
            pred_attention_mask = batch['pred']['attention_mask'].to(device)

            pred_labels = batch['label']['input_ids'].to(device)
            pred_outputs = student_model.generate(
                input_ids=pred_inputs, attention_mask=pred_attention_mask, max_new_tokens=32
            )

            expl_inputs = batch['expl']['input_ids'].to(device)
            expl_attention_mask = batch['expl']['attention_mask'].to(device)
            expl_outputs = student_model.generate(
                input_ids=expl_inputs, attention_mask=expl_attention_mask, max_new_tokens=32
            )

            val_preds.extend(tokenizer.batch_decode(pred_outputs, skip_special_tokens=True))
            val_labels.extend(tokenizer.batch_decode(pred_labels, skip_special_tokens=True))

            inputs_list.extend(tokenizer.batch_decode(pred_inputs, skip_special_tokens=True))
            expl_list.extend(tokenizer.batch_decode(expl_outputs, skip_special_tokens=True))

    val_acc = compute_equation_acc(val_preds, val_labels)

    for i, j, k in zip(inputs_list[:5], expl_list[:5], val_preds[:5]):
        print('prompt: {}:\nexplain: {}\nequation: {}\n\n'.format(i, j, k))

    print(f'Validation Accuracy epoch {epoch + 1}, Accuracy: {val_acc}')
    print('time: {:0.2f} seconds'.format(time.time() - start))

# TODO save the model