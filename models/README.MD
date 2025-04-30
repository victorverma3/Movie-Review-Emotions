# Models

## Best Ekman Emotion Classifier (EEC) Model

-   all-MiniLM-L6-v2 Embedding Model
-   Combined training dataset
-   Learning rate: 0.01
-   Momentum: 0.9
-   Epochs: 100
-   Batch size: 64
-   Mean F1-score: 0.563

## Best Training Parameter Configurations

The mean F1-score refers to the average of the test F1-score and the validation
F1-score.

### all-MiniLM-L6-v2 Embedding Model

Base Training Dataset

-   Learning rate: 0.05
-   Momentum: 0.5
-   Epochs: 100
-   Batch size: 64
-   Mean F1-score: 0.529

Combined Training Dataset

-   Learning rate: 0.01
-   Momentum: 0.9
-   Epochs: 100
-   Batch size: 64
-   Mean F1-score: 0.563

### m3e-base Embedding Model

Base Training Dataset

-   Learning rate: 0.01
-   Momentum: 0.5
-   Epochs: 100
-   Batch size:128
-   Mean F1-score: 0.467

Combined Training Dataset

-   Learning rate: 0.01
-   Momentum: 0.5
-   Epochs: 200
-   Batch size: 256
-   Mean F1-score: 0.503
