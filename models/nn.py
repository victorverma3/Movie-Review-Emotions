import argparse
import itertools
import json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Sequence, Tuple


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils import (
    load_emotion_data_embeddings,
    load_emotion_data_splits,
    evaluate_model,
    plot_training_losses,
)


class EkmanEmotionClassifer(nn.Module):
    def __init__(self, input_dim: int = 384, lr: float = 0.05, momentum: float = 0.9):

        super().__init__()
        self.input_dim = input_dim
        self.num_classes = 6
        self.hidden_size = 256
        self.linear1 = nn.Linear(
            in_features=self.input_dim, out_features=self.hidden_size
        )
        self.linear2 = nn.Linear(
            in_features=self.hidden_size, out_features=self.num_classes
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def predict(self, x: np.ndarray[float]) -> np.ndarray[int]:

        # Prepares input tensor
        x_tensor = torch.tensor(data=x, dtype=torch.float32)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset)

        # Gathers predictions
        predictions = []
        with torch.no_grad():
            for (inputs,) in tqdm(loader, desc="Predicting emotions"):
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def train(
        self,
        x_train: np.ndarray[float],
        y_train: Sequence[int],
        num_epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> Sequence[float]:

        # Prepares training data
        x_tensor = torch.tensor(data=x_train, dtype=torch.float32)
        y_tensor = torch.tensor(data=y_train, dtype=torch.long)

        trainset = TensorDataset(x_tensor, y_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        # Completes training iterations
        print("----------")
        losses = []
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            running_loss = 0.0
            for _, data in enumerate(trainloader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                # Updates model parameters
                outputs = self.forward(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Updates running loss
                running_loss += loss.item()

            # Aggregates loss over time
            losses.append(running_loss)

            if verbose:
                print(f"Epoch {epoch + 1} loss: {running_loss / 10:.3f}")

        return losses

    def evaluate(
        self,
        true: Sequence[int],
        pred: np.ndarray[int],
        data_variant: str,
        training_dataset: str,
        prediction_variant: str,
        save_metrics: bool = True,
        create_confusion_matrix: bool = True,
        verbose: bool = True,
    ) -> Tuple[float, float, float, float]:

        # Evaluates model
        accuracy, precision, recall, f1_score = evaluate_model(
            actual=true,
            predicted=pred,
            classification_report_save_path=f"../data/nn/{prediction_variant}/{training_dataset}/{data_variant}_nn_emotion_classification_report_{training_dataset}.json",
            confusion_matrix_title=f"NN Emotion Confusion Matrix ({data_variant.title()})",
            confusion_matrix_save_path=f"../data/nn/{prediction_variant}/{training_dataset}/{data_variant}_nn_emotion_confusion_matrix_{training_dataset}.png",
            save_metrics=save_metrics,
            create_confusion_matrix=create_confusion_matrix,
        )

        if verbose:
            print(f"{data_variant.title()} accuracy: {accuracy:.3f}")
            print(f"{data_variant.title()} precision: {precision:.3f}")
            print(f"{data_variant.title()} recall: {recall:.3f}")
            print(f"{data_variant.title()} f1 score: {f1_score:.3f}")

        return accuracy, precision, recall, f1_score

    def save(self, save_path: str) -> None:

        torch.save(obj=self.state_dict(), f=save_path)
        print(f"Saved model to {args.save_path}")


# Performs grid search over neural network parameters
def nn_grid_search(
    lr_options: Sequence[float],
    momentum_options: Sequence[float],
    num_epochs_options: Sequence[int],
    batch_size_options: Sequence[int],
    embedding_model_name: str,
    embedded_x_train: np.ndarray[float],
    embedded_x_test: np.ndarray[float],
    embedded_x_val: np.ndarray[float],
    y_train: Sequence[int],
    y_test: Sequence[int],
    y_val: Sequence[int],
    training_dataset: str,
) -> None:

    # Performs grid search
    for lr, momentum, num_epochs, batch_size in tqdm(
        itertools.product(
            lr_options, momentum_options, num_epochs_options, batch_size_options
        ),
        total=len(lr_options)
        * len(momentum_options)
        * len(num_epochs_options)
        * len(batch_size_options),
        desc="Performing grid search",
    ):
        # Trains classification model
        model = EkmanEmotionClassifer(
            input_dim=len(embedded_x_train[0]),
            lr=lr,
            momentum=momentum,
        )
        model.train(
            x_train=embedded_x_train,
            y_train=y_train,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=False,
        )

        # Evaluates test performance
        print("----------")
        pred_test = model.predict(x=embedded_x_test)
        test_accuracy, test_precision, test_recall, test_f1_score = model.evaluate(
            true=y_test,
            pred=pred_test,
            data_variant="test",
            training_dataset=training_dataset,
            prediction_variant="predictions",
            save_metrics=False,
            create_confusion_matrix=False,
            verbose=False,
        )

        # Evaluates validation performance
        print("----------")
        pred_val = model.predict(x=embedded_x_val)
        val_accuracy, val_precision, val_recall, val_f1_score = model.evaluate(
            true=y_val,
            pred=pred_val,
            data_variant="validation",
            training_dataset=training_dataset,
            prediction_variant="predictions",
            save_metrics=False,
            create_confusion_matrix=False,
            verbose=False,
        )

        # Aggregates metrics
        parameter_metrics = {
            "learning_rate": lr,
            "momentum": momentum,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1_score,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1_score,
        }

        # Saves parameter metrics
        with open(
            f"./nn_param_grid_search/{embedding_model_name}/{training_dataset}/metrics_lr{lr}_momentum{momentum}_epochs{num_epochs}_batch{batch_size}.json",
            "w",
        ) as f:
            json.dump(parameter_metrics, f, indent=4)


# Gets best neural network parameters from grid search results
def get_best_nn_parameters(
    embedding_model_name: str,
    training_dataset: str,
) -> Tuple[float, float, int, int, float]:

    directory = f"./nn_param_grid_search/{embedding_model_name}/{training_dataset}"
    grid_search_results = []

    # Concatenates grid search results
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                grid_search_results.append(json.load(f))

    # Retrieves best parameters
    best_lr = None
    best_momentum = None
    best_num_epochs = None
    best_batch_size = None
    best_f1 = 0

    for result in grid_search_results:
        mean_f1 = np.mean([result["test_f1_score"], result["val_f1_score"]])
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_lr = result["learning_rate"]
            best_momentum = result["momentum"]
            best_num_epochs = result["num_epochs"]
            best_batch_size = result["batch_size"]

    return best_lr, best_momentum, best_num_epochs, best_batch_size, best_f1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument(
        "-m",
        "--mode",
        help="The neural network mode",
        choices=["train", "search", "retrieve"],
        default="train",
    )

    # Training dataset
    parser.add_argument(
        "-td",
        "--training-dataset",
        help="Specifies the training dataset.",
        choices=["base", "combined"],
        default="combined",
    )

    # Embedding model name
    parser.add_argument(
        "-e",
        "--embedding_model_name",
        help="Specifies the training dataset.",
        choices=["all-MiniLM-L6-v2", "m3e-base"],
        default="all-MiniLM-L6-v2",
    )

    # Plot losses
    parser.add_argument(
        "-pl",
        "--plot_losses",
        help="Specifies if the training losses should be plotted.",
        action="store_true",
    )

    # Save path
    parser.add_argument(
        "-sp", "--save_path", help="Defines the model save path.", default=None
    )

    args = parser.parse_args()

    if args.mode == "train":
        # Loads emotion data splits
        _, y_train, _, y_test, _, y_val = load_emotion_data_splits(
            training_dataset=args.training_dataset
        )

        # Loads embeddings
        embedded_x_train, embedded_x_test, embedded_x_val = (
            load_emotion_data_embeddings(
                training_dataset=args.training_dataset,
                embedding_model_name=args.embedding_model_name,
            )
        )

        # Sets training parameters
        if args.embedding_model_name == "all-MiniLM-L6-v2":
            if args.training_dataset == "base":
                lr = 0.05
                momentum = 0.5
                num_epochs = 100
                batch_size = 64
            elif args.training_dataset == "combined":
                lr = 0.01
                momentum = 0.9
                num_epochs = 100
                batch_size = 64
        elif args.embedding_model_name == "m3e-base":
            if args.training_dataset == "base":
                lr = 0.05
                momentum = 0.5
                num_epochs = 100
                batch_size = 64
            elif args.training_dataset == "combined":
                lr = 0.01
                momentum = 0.5
                num_epochs = 200
                batch_size = 256

        # Trains classification model
        model = EkmanEmotionClassifer(
            input_dim=len(embedded_x_train[0]),
            lr=lr,
            momentum=momentum,
        )
        losses = model.train(
            x_train=embedded_x_train,
            y_train=y_train,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        # Plots losses
        if args.plot_losses:
            plot_training_losses(
                losses=losses,
                save_path=f"../data/nn/figures/{args.training_dataset}_training_loss.png",
            )

        # Evaluates test performance
        print("----------")
        pred_test = model.predict(x=embedded_x_test)
        model.evaluate(
            true=y_test,
            pred=pred_test,
            data_variant="test",
            training_dataset=args.training_dataset,
            prediction_variant="predictions",
        )

        # Evaluates validation performance
        print("----------")
        pred_val = model.predict(x=embedded_x_val)
        model.evaluate(
            true=y_val,
            pred=pred_val,
            data_variant="validation",
            training_dataset=args.training_dataset,
            prediction_variant="predictions",
        )

        # Saves model
        if args.save_path is not None:
            print("----------")
            model.save(save_path=args.save_path)

    elif args.mode == "search":
        # Loads emotion data splits
        _, y_train, _, y_test, _, y_val = load_emotion_data_splits(
            training_dataset=args.training_dataset
        )

        # Loads embeddings
        embedded_x_train, embedded_x_test, embedded_x_val = (
            load_emotion_data_embeddings(
                training_dataset=args.training_dataset,
                embedding_model_name=args.embedding_model_name,
            )
        )

        # Configures grid search parameter options
        lr_options = [0.01, 0.05, 0.1, 0.25]
        momentum_options = [0.5, 0.9]
        num_epochs_options = [100, 200]
        batch_size_options = [64, 128, 256]

        # Performs grid search
        nn_grid_search(
            lr_options=lr_options,
            momentum_options=momentum_options,
            num_epochs_options=num_epochs_options,
            batch_size_options=batch_size_options,
            embedding_model_name=args.embedding_model_name,
            embedded_x_train=embedded_x_train,
            embedded_x_test=embedded_x_test,
            embedded_x_val=embedded_x_val,
            y_train=y_train,
            y_test=y_test,
            y_val=y_val,
            training_dataset=args.training_dataset,
        )
    elif args.mode == "retrieve":
        # Gets best neural network parameters
        learning_rate, momentum, num_epochs, batch_size, f1 = get_best_nn_parameters(
            embedding_model_name=args.embedding_model_name,
            training_dataset=args.training_dataset,
        )
        print("Embedding model name:", args.embedding_model_name)
        print("Training dataset:", args.training_dataset)
        print("Best learning rate:", learning_rate)
        print("Best momentum:", momentum)
        print("Best num epochs:", num_epochs)
        print("Best batch size:", batch_size)
        print("Best mean of test and validation f1 score:", f1)
