import transformers
import accelerate

# Load the pre-trained model
model = transformers.AutoModel.from_pretrained("facebook/bart-base")

# Create an Accelerate Accelerator
accelerator = accelerate.Accelerator()

# Move the model to the GPU
model.to(accelerator.device)

import csv

def load_dataset(csv_file):
    """Loads a code migration dataset from a CSV file.

    Args:
        csv_file: The path to the CSV file.

    Returns:
        A list of tuples, where each tuple contains the input and output code snippets.
    """

    dataset = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            input_code = row[0]
            output_code = row[1]

            dataset.append((input_code, output_code))

    return dataset

# Load the modified dataset
dataset = load_dataset("/content/code_to_code_geekforgeek.csv")
print(type(dataset))

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

def create_train_dataloader(dataset, batch_size):
    """Creates a training dataloader for a code migration dataset.

    Args:
        dataset: A list of tuples, where each tuple contains the input and output code snippets.
        batch_size: The batch size.

    Returns:
        A DataLoader object.
    """

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader

# Create the training dataloader
train_dataloader = create_train_dataloader(dataset, batch_size=64)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=1e-5)

from transformers import get_linear_schedule_with_warmup
import torch

num_training_steps = len(train_dataloader) * 20

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)

# Start training
for epoch in range(2000):
    for batch in train_dataloader:
        input_text = ""
        for snippet_tuple in batch:
          if snippet_tuple:
            # print(len(snippet_tuple))
            input_text += snippet_tuple[0] + "\n" + snippet_tuple[1] + "\n"
        # Convert the list of code snippets to tensors
        batch = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        # Move the batch to the GPU
        batch = batch.to(accelerator.device)
        # print("To forward pass")
        # Forward pass
        outputs = model(**batch)

        # Loss
        # print("To loss")
        loss = torch.mean(outputs[0])  # Compute the average loss across the batch.

        # Backward pass
        # print("To backward pass")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate
        # print("Updating learning rate")
        scheduler.step()
    print(f"Epoch {epoch}, Batch loss: {loss.item()}")  # Print the current loss

# Save the fine-tuned model
model.save_pretrained("fine-tuned-bart-base-code-migration")
