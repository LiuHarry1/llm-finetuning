import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# Define a dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Define the model class
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 64)  # Hidden state
        c_0 = torch.zeros(1, x.size(0), 64)  # Cell state
        _, (h_n, _) = self.lstm(x.unsqueeze(1), (h_0, c_0))
        out = self.fc(h_n[-1])
        return out


def train_and_save_model(model_path: str):
    # Sample dataset
    texts = ['I love this movie', 'This movie is terrible', 'Best film ever', 'Worst film ever']
    labels = ['positive', 'negative', 'positive', 'negative']

    # Encode the labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Vectorize the text
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()
    y = torch.tensor(encoded_labels)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoader
    train_dataset = TextDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Define model, loss function, and optimizer
    model = LSTMClassifier(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(5):  # Use more epochs for actual training
        model.train()
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/5, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


from peft import LoraConfig, get_peft_model

def fine_tune_with_lora(model_path: str):
    # Sample dataset for fine-tuning (can be different from the initial dataset)
    texts = ['Amazing movie', 'Terrible film']
    labels = ['positive', 'negative']

    # Encode the labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Vectorize the text
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()
    y = torch.tensor(encoded_labels)

    # Create DataLoader
    fine_tune_dataset = TextDataset(X, y)
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=2, shuffle=True)

    # Define model architecture and load pretrained weights
    model = LSTMClassifier(input_dim=X.shape[1], hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=["lstm"],  # Specify which layers to adapt
        lora_dropout=0.1,
    )

    # Integrate LoRA into the model
    model = get_peft_model(model, lora_config)

    # Fine-tune the model with LoRA
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # Use more epochs for actual fine-tuning
        model.train()
        for texts, labels in fine_tune_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
        print(f"Fine-tuning Epoch {epoch+1}/5, Loss: {loss.item()}")

if __name__ == '__main__':
    # Train and save the model
    train_and_save_model('text_classification_model.pth')

    # Fine-tune the model with LoRA
    # fine_tune_with_lora('text_classification_model.pth')
