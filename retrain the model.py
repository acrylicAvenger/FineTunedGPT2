import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Load the pre-trained model and tokenizer
output_dir = "/savedModel"
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Load and tokenize the new dataset
with open("C:\\Users\\Darshan\\Desktop\\books1\\mergedfiles14.txt", 'r', encoding='utf-8') as file:
    new_text = file.read()

# Tokenize the new dataset
new_input_ids = tokenizer.encode(new_text, add_special_tokens=True, max_length=512, pad_to_max_length=True, truncation=True, return_tensors='pt')

# Create DataLoader for new dataset
class NewDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

batch_size = 8
new_dataset = NewDataset(new_input_ids)
new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the fine-tuning parameters
epochs = 10
learning_rate = 1e-4
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Fine-tuning loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(new_dataloader, desc=f'Epoch {epoch + 1}', leave=False)
    for batch in progress_bar:
        inputs = batch.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({'loss': total_loss / len(progress_bar)})

    # Print average loss at the end of each epoch
    print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(progress_bar)}")

# Save the newly trained model in the same directory as the saved model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training completed. Model saved in the same directory as the saved model.")
