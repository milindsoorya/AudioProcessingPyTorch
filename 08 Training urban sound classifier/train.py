import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagete loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("----------------")
    print("Training is done.")


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/Users/milindsoorya/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/Users/milindsoorya/datasets/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Instatnitaning dataset object
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                            SAMPLE_RATE, NUM_SAMPLES, device)
    
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)
     
    # build model
    cnn = CNNNetwork().to(device)

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), 
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and saved at cnn.pth")