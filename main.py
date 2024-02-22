import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split


# C:\Users\Darshan\Desktop\myFineTunedGPT saved model and api path


from transformers import GPT2LMHeadModel, GPT2Config, AdamW
from tqdm import tqdm
from transformers import AdamW

with open("C:\\Users\\Darshan\\Desktop\\books1\\mergedfiles1.txt", 'r', encoding='utf-8') as file:
    text = file.read()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data = train_test_split(text, test_size=0.2, random_state=42)

# tokenization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Encode and tokenize the training data
train_input_ids = tokenizer.encode(train_data, add_special_tokens=True, max_length=512, pad_to_max_length=True,
                                   truncation=True, return_tensors='pt')

# Encode and tokenize the testing data
test_input_ids = tokenizer.encode(test_data, add_special_tokens=True, max_length=512, pad_to_max_length=True,
                                  truncation=True, return_tensors='pt')

# Step 4: Create DataLoader
class ShakespeareDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


# Create DataLoader for training data
train_dataset = ShakespeareDataset(train_input_ids)
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create DataLoader for testing data
test_dataset = ShakespeareDataset(test_input_ids)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 5: Fine-tuning the model
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model_path = '/huggingFaceGPT2'
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=model_path, config=config)
model.to(device)

# Set the fine-tuning parameters
epochs = 10
learning_rate = 1e-4

optimizer = AdamW(model.parameters(), lr=learning_rate)


# Fine-tuning loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', leave=False)
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


# Evaluation on the test set
model.eval()
total_eval_loss = 0

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Evaluation', leave=False):
        inputs = batch.to(device)

        outputs = model(inputs, labels=inputs)
        loss = outputs.loss

        total_eval_loss += loss.item()

    average_eval_loss = total_eval_loss / len(test_dataloader)
    print(f"Average Evaluation Loss: {average_eval_loss}")


# Define the path to save the model
output_dir = "./savedModel"

# Save the model
model.save_pretrained(output_dir)

# Save the tokenizer
tokenizer.save_pretrained(output_dir)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch

# Load the saved model and tokenizer
output_dir = "/savedModel"
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Define the prompts
prompts = ["Corn for the rich men only: with these shreds"]


# Generate output for each prompt
for prompt in prompts:
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(encoded_prompt).to(device)

    # Generate text
    output = model.generate(encoded_prompt, max_length=100, num_return_sequences=1, attention_mask=attention_mask)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Prompt:", prompt)
    print("Generated Text:", generated_text)
    print()
