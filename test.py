from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from BiLSTM import BiLSTM
from ner_dataset import NERDataset, num_chars, num_labels


def model_test(
    model: nn.Module,
    test_loader: DataLoader,
    test_dataset: NERDataset,
):
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            _, preds = model(inputs, labels)

            num_correct += (preds == labels).sum().item()

    print(f"Accuracy: {num_correct / len(test_dataset) * 100:.4f}%")


if __name__ == "__main__":
    FILE = Path(__file__).parent / "model.pth"
    model = BiLSTM(
        vocab_size=num_chars,
        embed_size=128,
        hidden_size=128,
        output_size=num_labels,
    )
    model.load_state_dict(torch.load(FILE))

    # test dataset
    test_data_file = Path("data/test.txt")
    test_dataset = NERDataset(test_data_file)

    # test dataloader: batch_size=1 for simplicity (avoid padding and truncating)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    
    # Model test
    model_test(model, test_loader, test_dataset)
