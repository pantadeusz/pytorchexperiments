# The skeleton was based on the PyTorch documentation, but right now it is almost completely new code

# Tadeusz Puźniakowski, 2024


import sys
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,(3,3)), # 24*24*32
            nn.ReLU(),
            nn.MaxPool2d(2), # 12 * 12* 32
            nn.Flatten(),
            nn.Linear(13*13*32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 26*26*32),
            nn.ReLU(),
            nn.Unflatten(1,(32,26,26)),
            nn.ConvTranspose2d(32,1,(3,3))
        )

    def forward(self, x):
        logits = self.encoder(x)
        x = self.decoder(logits)
        return x


def train(epoch, dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        print(f"[{epoch}] | Batch: {batch} | Loss: {loss:>7f}      ",end="\r")
    print("")

def save_images(model, device, raw_training_data):
    eval_model = model.eval() # end of training
 
    with torch.no_grad():
        i = 0
        for test_row in raw_training_data:
            i = i + 1
            x, y = test_row[0], test_row[1]
            x = torch.reshape(x, (1,1,28,28))
            x = x.to(device)
            pred = eval_model(x)
            save_image(pred,f'results/mnist_{i}_y.png')
            save_image(x,f'results/mnist_{i}_x.png')
            if i % 1000 == 0: print(str(i) + " / " + str(len(test_data)))
            if (i>20):
                return


def main() -> int:
    print("PyTorch: ",torch.__version__)
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 10000

    raw_training_data = [(a,a) for a,_ in training_data ]
    train_dataloader = DataLoader(raw_training_data, batch_size=batch_size, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    try:
        model.load_state_dict(torch.load("modelautoenc.pth", weights_only=True))
    except:
        print("could not load pretrained model")

    loss_fn = nn.MSELoss()

    epochs = 30
    lr = 0.001
    start = time.time()
    train_dataloader_ = [ (X.to(device), y.to(device)) for batch, (X, y) in enumerate(train_dataloader) ]
    for t in range(epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train(t, train_dataloader_, model, loss_fn, optimizer, device)
        lr = lr * 0.9
        torch.save(model.state_dict(), "modelautoenc.pth")
    end = time.time()
    print("Training time[s]: ",end - start)

    save_images(model, device, raw_training_data)


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
