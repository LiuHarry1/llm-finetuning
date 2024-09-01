import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torchviz import make_dot
from torchsummary import summary
# Define a dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Define the fully connected network model
class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import joblib  # For saving the vectorizer

def train_and_save_model(model_path: str, vectorizer_path: str):
    # Sample dataset
    texts = ['I love this movie', 'greate movie', 'This movie is terrible', 'Best film ever', 'Worst film ever']
    labels = ['positive', 'positive',  'negative', 'positive', 'negative']

    # Encode the labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Vectorize the text
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()
    y = torch.tensor(encoded_labels)

    # Save the vectorizer
    joblib.dump(vectorizer, vectorizer_path)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoader
    train_dataset = TextDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Define model, loss function, and optimizer
    model = FullyConnectedClassifier(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(5):  # Use more epochs for actual training
        model.train()
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze()  # Adjust output to match target shape
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/5, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")
    print(model)
    summary(model, input_size=(X_train.shape[1],))

    # # Create a dummy input tensor for visualization (e.g., batch size of 1)
    # dummy_input = torch.randn(1, X_train.shape[1])
    #
    # # Forward pass with the dummy input to get the computational graph
    # output = model(dummy_input)
    #
    # # Generate the visualization using torchviz
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.render("saved_model_visualization", format="png")


# Train and save the model and vectorizer
train_and_save_model('fully_connected_model.pth', 'vectorizer.pkl')

from peft import LoraConfig, get_peft_model

def fine_tune_with_lora(model_path: str, vectorizer_path: str):
    # Load the saved vectorizer
    vectorizer = joblib.load(vectorizer_path)

    # Sample dataset for fine-tuning (can be different from the initial dataset)
    texts = ['Amazing movie', 'Terrible film']
    labels = ['positive', 'negative']

    # Encode the labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Use the loaded vectorizer to transform the text
    X = vectorizer.transform(texts).toarray()
    y = torch.tensor(encoded_labels)

    # Create DataLoader
    fine_tune_dataset = TextDataset(X, y)
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=2, shuffle=True)

    # Define model architecture and load pretrained weights
    model = FullyConnectedClassifier(input_dim=X.shape[1], hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=["fc1"],  # Specify which layers to adapt
        lora_dropout=0.1,
    )

    # Integrate LoRA into the model
    model = get_peft_model(model, lora_config)

    print(model)
    summary(model, input_size=(X.shape[1],))

    # # Create a dummy input tensor for visualization (e.g., batch size of 1 and input_dim size)
    # dummy_input = torch.randn(1, X.shape[1])  # X.shape[1] should match the input_dim of your model
    #
    # # Forward pass with the dummy input to get the computational graph
    # output = model(dummy_input)
    #
    # # Generate the visualization using torchviz
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.render("peft_model_visualization", format="png")

    # Fine-tune the model with LoRA
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # Use more epochs for actual fine-tuning
        model.train()
        for texts, labels in fine_tune_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze()  # Adjust output to match target shape
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        print(f"Fine-tuning Epoch {epoch+1}/5, Loss: {loss.item()}")

# Fine-tune the model with LoRA
fine_tune_with_lora('fully_connected_model.pth', 'vectorizer.pkl')


