import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Conditional VAE model (same as training)
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.fc1 = nn.Linear(28*28 + num_classes, 256)
        self.fc21 = nn.Linear(256, latent_dim)
        self.fc22 = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim + num_classes, 256)
        self.fc4 = nn.Linear(256, 28*28)
        self.num_classes = num_classes

    def encode(self, x, labels):
        labels_onehot = torch.eye(self.num_classes, device=device)[labels]
        x = torch.cat([x.view(-1, 28*28), labels_onehot], dim=1)
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        labels_onehot = torch.eye(self.num_classes, device=device)[labels]
        z = torch.cat([z, labels_onehot], dim=1)
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

# Device
device = torch.device('cpu')

# Load the trained CVAE model
model = ConditionalVAE().to(device)
model.load_state_dict(torch.load('cvae_mnist_100.pth', map_location=device))
model.eval()

# Streamlit App
st.title("Conditional VAE MNIST Digit Generator 🤖✍️")

digit = st.number_input("Enter Digit to Generate (0-9):", min_value=0, max_value=9, step=1)

if st.button("Generate 5 Images"):
    with torch.no_grad():
        z = torch.randn(5, 20).to(device)  # 5 random latent vectors
        labels = torch.tensor([digit] * 5).to(device)
        samples = model.decode(z, labels).cpu()
        samples = samples.view(5, 28, 28)

        # Plot and show
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(samples[i], cmap='gray')
            axes[i].axis('off')
        st.pyplot(fig)
