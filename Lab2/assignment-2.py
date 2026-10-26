from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_utils.vinafood21_dataset import VinaFood21
from models.googlenet import GoogLeNet

CHECKPOINT = Path("./checkpoints/assignment-2/best_model.pth")
NUM_CLASSES = 21
device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions = []
    trues = []

    with torch.no_grad():
        for item in tqdm(dataloader, desc="Evaluating"):
            images = item["image"].to(device)
            labels = item["label"].to(device)

            logits = model(images)
            predicted_labels = logits.argmax(dim=1)

            predictions.extend(predicted_labels.cpu().tolist())
            trues.extend(labels.cpu().tolist())


    metric_options = {
        "labels": list(range(NUM_CLASSES)),
        "average": "macro",
        "zero_division": 0,
    }
    return {
        "accuracy": accuracy_score(trues, predictions),
        "precision": precision_score(trues, predictions, **metric_options),
        "recall": recall_score(trues, predictions, **metric_options),
        "f1": f1_score(trues, predictions, **metric_options),
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    EPOCHS = 10
    BATCH_SIZE = 16

    train_dataset = VinaFood21("D:/Documents/DS201 - TH/Lab2/dataset/VinaFood21/train")
    test_dataset = VinaFood21(
        "D:/Documents/DS201 - TH/Lab2/dataset/VinaFood21/test",
        class_to_idx=train_dataset.class_to_idx,
    )

    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))),
        test_size=0.2,
        random_state=42,
        stratify=[item["label"] for item in train_dataset._data],
    )
    train_dataloader = DataLoader(
        dataset=Subset(train_dataset, train_indices),
        batch_size=16,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=Subset(train_dataset, val_indices),
        batch_size=16,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
    )

    model = GoogLeNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Train: {len(train_indices)} | Validation: {len(val_indices)} | "
        f"Test: {len(test_dataset)}"
    )
    print(f"Classes: {train_dataset.class_to_idx}")
    best_score = -1.0
    best_score_name = "f1"

    for epoch in range(EPOCHS):
        losses = []
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training") as pb:
            for item in pb:
                image = item["image"].to(device)
                label = item["label"].to(device)

                # Forward pass
                output = model(image)
                loss = loss_fn(output, label)
                losses.append(loss.item())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pb.set_postfix({"loss": np.array(losses).mean()})

        scores = evaluate(val_dataloader, model)
        print("Validation:")
        for score in scores:
            print(f"\t- {score}: {scores[score]}")

        current_score = scores[best_score_name]
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"Saved checkpoint | F1-macro: {best_score:.4f}")

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=device, weights_only=True)
    )
    print("Test:", evaluate(test_dataloader, model))
    print(f"Best checkpoint: {CHECKPOINT}")
