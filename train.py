import torch
import numpy as np
import time
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader


def tokenize_function(examples):
    # Assuming that your data has 'problem', 'explain', 'formula', and 'reason' as the keys
    problem_encodings = tokenizer(
        "predict: " + examples['problem'], truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    expl_encodings = tokenizer(
        "explain: " + examples['problem'], truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    formula_encodings = tokenizer(
        examples['formula'], truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    reasoning_encodings = tokenizer(
        examples['reason'], truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )

    return {
        'problem': problem_encodings,
        'explain': expl_encodings,
        'formula': formula_encodings,
        'reason': reasoning_encodings
    }


def collate_function(batch):
    def stack_encoding(key):
        return {
            'input_ids': torch.stack([torch.tensor(item[key]['input_ids']).squeeze() for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item[key]['attention_mask']).squeeze() for item in batch]),
        }

    def stack_tensors(key):
        return torch.stack([torch.tensor(item[key]['input_ids']).squeeze() for item in batch])

    return {
        'problem': stack_encoding('problem'),
        'explain': stack_encoding('explain'),
        'formula': {'input_ids': stack_tensors('formula')},
        'reason': {'input_ids': stack_tensors('reason')},
    }


def get_training_example(batch, key, device):
    inputs = batch[key]['input_ids'].to(device)
    attn_mask = batch[key]['attention_mask'].to(device)
    return inputs, attn_mask


def eval_formula(formula):
    try:
        answer = eval(formula)
    except:
        answer = np.nan
    return answer


def compute_formula_acc(problems, formulas):
    problems = [eval_formula(problem) for problem in problems]
    formulas = [eval_formula(formula) for formula in formulas]
    return np.mean(np.array(problems) == np.array(formulas))


# Checking for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')

tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')
student_model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base')
student_model = student_model.to(device)

dataset = load_dataset('json', data_files='data_train.json', split='train')

# Split the dataset
train_val_dataset = dataset.train_test_split(test_size=0.1)

# Apply the tokenize function to all examples in the dataset
tokenized_train_dataset = train_val_dataset["train"].map(tokenize_function)
tokenized_val_dataset = train_val_dataset["test"].map(tokenize_function)

BATCH_SIZE = 8

# Create DataLoaders to handle batching
train_loader = DataLoader(tokenized_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_function)
val_loader = DataLoader(tokenized_val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_function)

num_epochs = 200
accumulation_steps = 8
total_data_points = len(tokenized_train_dataset)
effective_batch_size = BATCH_SIZE * accumulation_steps  # The effective batch size
total_iterations = int((total_data_points / effective_batch_size) * num_epochs)  # Recalculate total_iterations

print("num iterations: ", total_iterations)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)
alpha = 0.5  # Assume an equal weight for simplicity, adjust as needed
start = time.time()
save_directory = 'checkpoints'
save_every = 20

# Training loop
for epoch in range(num_epochs):
    student_model.train()
    running_loss = 0.0
    current_lr = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        pred_inputs, pred_attention_mask = get_training_example(batch, 'problem', device)
        pred_labels, _ = get_training_example(batch, 'formula', device)

        expl_inputs, expl_attention_mask = get_training_example(batch, 'explain', device)
        expl_labels, _ = get_training_example(batch, 'reason', device)

        # Forward pass for predicting the result as a formula
        pred_outputs = student_model(input_ids=pred_inputs, attention_mask=pred_attention_mask, labels=pred_labels)

        # Forward pass for explaining the result
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

            current_lr = optimizer.param_groups[0]['lr']

    print(f'Current learning rate: {current_lr}')
    print(f'Training loss epoch {epoch + 1}: {running_loss / len(train_loader)}')
    print('time: {:0.2f} seconds'.format(time.time() - start))

    # Save the model and evaluate the model every few epochs
    if epoch == 0 or (epoch + 1) % save_every == 0:
        # Evaluation
        student_model.eval()
        val_preds, val_labels = [], []
        inputs_list, expl_list = [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                pred_inputs, pred_attention_mask = get_training_example(batch, 'problem', device)
                pred_labels, _ = get_training_example(batch, 'formula', device)

                expl_inputs, expl_attention_mask = get_training_example(batch, 'explain', device)
                expl_labels, _ = get_training_example(batch, 'reason', device)

                pred_outputs = student_model.generate(
                    input_ids=pred_inputs, attention_mask=pred_attention_mask, max_new_tokens=32
                )

                expl_outputs = student_model.generate(
                    input_ids=expl_inputs, attention_mask=expl_attention_mask, max_new_tokens=32
                )

                val_preds.extend(tokenizer.batch_decode(pred_outputs, skip_special_tokens=True))
                val_labels.extend(tokenizer.batch_decode(pred_labels, skip_special_tokens=True))

                inputs_list.extend(tokenizer.batch_decode(pred_inputs, skip_special_tokens=True))
                expl_list.extend(tokenizer.batch_decode(expl_outputs, skip_special_tokens=True))

        val_acc = compute_formula_acc(val_preds, val_labels)

        for i, j, k in zip(inputs_list[:8], expl_list[:8], val_preds[:8]):
            print('prompt: {}\nthought: {}\nresult: {}\n\n'.format(i.split('predict: ')[1], j, k))

        print(f'Validation Accuracy epoch {epoch + 1}, Accuracy: {val_acc}')
        print('time: {:0.2f} seconds'.format(time.time() - start))

        tokenizer.save_pretrained(save_directory)
        student_model.save_pretrained(save_directory)
