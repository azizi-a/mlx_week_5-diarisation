import torch
import whisper
import editdistance
import numpy as np
import wandb

from tqdm import tqdm

from data.ami import AMIDataset, dataset
from torch.utils.data import DataLoader

# Load the dataset
ds = AMIDataset(dataset)

# Split dataset into train and test sets
train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
ds_train, ds_test = torch.utils.data.random_split(ds, [train_size, test_size])

data_loader_train = DataLoader(ds_train, batch_size=16, num_workers=1)
data_loader_validate = DataLoader(ds_test, batch_size=16, num_workers=1)

# Load the model
model = whisper.load_model("tiny.en")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


#
#
#
def forward_pass(model, text, audio):
  tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
  start_token = tokenizer.sot
  pad_token = tokenizer.eot

  # Convert audio to mel spectrogram
  audio = whisper.pad_or_trim(audio)
  audio = np.array(audio, dtype=np.float32)
  mel = whisper.log_mel_spectrogram(audio).to(device)

  # Encode all texts in batch
  target_ids = [tokenizer.encode(t, allowed_special={"<|startoflm|>"}) for t in text]
  # Convert to padded tensor using pad_sequence
  target_ids = [torch.tensor(ids, dtype=torch.long, device=device) for ids in target_ids]
  target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token).to(device)
  # Create the input tokens
  start_token_tensor = torch.tensor([start_token], dtype=torch.long, device=device).repeat(len(text), 1)
  input_tks = torch.cat([start_token_tensor, target_ids], dim=1)

  # Forward pass
  predictions = model(tokens=input_tks, mel=mel)
  input_removed_sot = input_tks[:, 1:].to(device)
  predictions = predictions[:, :-1, :]

  return predictions, input_removed_sot


#
#
#
def validate_model(model, data_loader_validate):
  model.eval()
  criterion = torch.nn.CrossEntropyLoss()

  wers = []
  losses = []

  for i, (audio, text) in enumerate(tqdm(data_loader_validate, desc="Evaluating")):
    predictions, input_removed_sot = forward_pass(model, text, audio)

    # Get predicted tokens and convert to text
    predicted_tokens = predictions.argmax(dim=-1)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    predicted_texts = [tokenizer.decode(t) for t in predicted_tokens]

    loss = criterion(predictions.transpose(1, 2), input_removed_sot)
    losses.append(loss.item())
    wandb.log({"Validation Loss": loss.item()})

    if i == 3:
      print("\nGround truth:")
      print(text[3])
      print("\nPredicted:")
      print(predicted_texts[3])

    # Calculate word error rate
    for prd_text, target_text in zip(predicted_texts, text):
      # Split into words
      prd_words = prd_text.split()
      target_words = target_text.split()

      # Calculate Levenshtein distance
      distance = editdistance.eval(prd_words, target_words)

      # Calculate WER
      if len(target_words) > 0:
        wer = distance / len(target_words)
        wers.append(wer)
        wandb.log({"WER": wer})

  average_loss = sum(losses) / len(losses)
  print(f"Average loss: {average_loss}")
  average_wer = sum(wers) / len(wers)
  print(f"Average WER: {average_wer * 100:.2f}%")
  wandb.log({"Average Validation Loss": average_loss, "Average WER": average_wer})


#
#
#
def train_model(model, data_loader_train, epoch=0):
  model.train()

  # Define the optimizer and criterion
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
  criterion = torch.nn.CrossEntropyLoss()

  for audio, text in tqdm(data_loader_train, desc="Training epoch"):
    # Forward pass
    predictions, input_removed_sot = forward_pass(model, text, audio)

    # Backward pass
    optimizer.zero_grad()
    loss = criterion(predictions.transpose(1, 2), input_removed_sot)
    loss.backward()
    optimizer.step()

    wandb.log({"Training Loss": loss.item()})

  print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
  wandb.log({"Epoch Loss": loss.item()})


#
#
#
# Initialize wandb
wandb.init(project="diarisation")

# zero shot validation
validate_model(model, data_loader_validate)

# train model
for epoch in range(8):
  train_model(model, data_loader_train, epoch)
  validate_model(model, data_loader_validate)

  artifact = wandb.Artifact(f"model-epoch-{epoch}", type="model")
  torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
  artifact.add_file(f"model_epoch_{epoch}.pt")
  wandb.log_artifact(artifact)

  wandb.finish()
