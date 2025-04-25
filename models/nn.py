import argparse
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Sequence


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import load_emotion_data_embedings, load_emotion_data_splits, evaluate_model


class EkmanEmotionClassifer(nn.Module):
    def __init__(self, input_dim: int, lr: float = 0.01, momentum: float = 0.9):

        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=6)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def train(self, x_train: np.ndarray, y_train: Sequence[int], num_epochs: int = 2):

        # Prepares training data
        x_tensor = torch.tensor(data=x_train, dtype=torch.float32)
        y_tensor = torch.tensor(data=y_train, dtype=torch.long)

        trainset = TensorDataset(x_tensor, y_tensor)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

        # Completes training iterations
        for epoch in range(num_epochs):

            running_loss = 0.0
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, labels = data

                self.optimizer.zero_grad()

                # Updates model parameters
                outputs = self.forward(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Updates running loss
                running_loss += loss.item()
                # if batch_idx % 100 == 0:
                #     print(
                #         f"Epoch {epoch + 1}, batch {batch_idx + 1} loss: {running_loss / 10:.3f}"
                #     )
                #     running_loss = 0.0
            print(f"Epoch {epoch + 1} loss: {running_loss / 10:.3f}")

        print("Finished training Ekman Emotion Classifier")

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, embedding_model_name: str, data_variant: str
    ):

        # Prepares testing data
        x_tensor = torch.tensor(data=x, dtype=torch.float32)
        y_tensor = torch.tensor(data=y, dtype=torch.long)

        testset = TensorDataset(x_tensor, y_tensor)
        testloader = DataLoader(testset)

        true = []
        pred = []

        # Evaluates accuracy
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = self.forward(inputs)
                _, predictions = torch.max(outputs, 1)

                true.extend(labels.cpu().numpy())
                pred.extend(predictions.cpu().numpy())

        # Evaluates model
        evaluate_model(
            actual=true,
            predicted=pred,
            classification_report_save_path=f"../data/nn/predictions/{embedding_model_name}/{data_variant}_nn_emotion_classification_report.json",
            confusion_matrix_title=f"BoW NN Emotion Confusion Matrix ({data_variant.title()})",
            confusion_matrix_save_path=f"../data/nn/predictions/{embedding_model_name}/{data_variant}_nn_emotion_confusion_matrix.png",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Embedding model
    parser.add_argument(
        "-embed",
        "--embedding-model-name",
        help="The vector embedding model.",
        choices=["bow", "all-MiniLM-L6-v2"],
        default="all-MiniLM-L6-v2",
    )

    args = parser.parse_args()

    # Loads emotion data splits
    _, y_train, _, y_test, _, y_val = load_emotion_data_splits()

    # Loads embeddings
    embedded_x_train, embedded_x_test, embedded_x_val = load_emotion_data_embedings(
        embedding_model_name=args.embedding_model_name
    )

    # Trains classification model
    model = EkmanEmotionClassifer(
        input_dim=len(embedded_x_train[0]), lr=0.05, momentum=0.9
    )
    model.train(x_train=embedded_x_train, y_train=y_train, num_epochs=50)

    # Evaluates classification model
    model.evaluate(
        x=embedded_x_test,
        y=y_test,
        embedding_model_name=args.embedding_model_name,
        data_variant="test",
    )
    model.evaluate(
        x=embedded_x_val,
        y=y_val,
        embedding_model_name=args.embedding_model_name,
        data_variant="validation",
    )
