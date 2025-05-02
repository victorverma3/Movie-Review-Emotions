# Analyzing Emotions in Reviews for Top-Rated Movies on Letterboxd

## Table of Contents

-   [Project Directory](#project-directory)
-   [Code Functionality](#code-functionality)

## Project Directory

The following subdirectories, listed in the project directory, are not uploaded
to GitHub due to the size of the files.

-   /data/nn/embeddings
-   /data/nn/figures
-   /data/processed
-   /data/nn/predictions
-   /data/output/embeddings
-   /data/output/predicted_movie_review_emotions.csv
-   /data/raw

```
data
|__ baseline
    |__ baseline_emotion_classification_report_combined.json
    |__ baseline_emotion_classification_report.json
    |__ baseline_emotion_confusion_matrix_combined.png
    |__ baseline_emotion_confusion_matrix.png
    |__ baseline_emotion_predictions_combined.csv
    |__ baseline_emotion_predictions.csv
|__ nn
    |__ best_predictions
        |__ base
            |__ test_nn_emotion_classification_report_base.json
            |__ test_nn_emotion_confusion_matrix_base.png
            |__ validation_nn_emotion_classification_report_base.json
            |__ validation_nn_emotion_confusion_matrix_base.png
        |__ combined
            |__ test_nn_emotion_classification_report_base.json
            |__ test_nn_emotion_confusion_matrix_base.png
            |__ validation_nn_emotion_classification_report_base.json
            |__ validation_nn_emotion_confusion_matrix_base.png
    |__ embeddings
        |__ all-MiniLM-L6-v2
            |__ x_test_combined_embeddings.npy
            |__ x_test_embeddings.npy
            |__ x_train_combined_embeddings.npy
            |__ x_train_embeddings.npy
            |__ x_val_combined_embeddings.npy
            |__ x_val_embeddings.npy
        |__ m3e-base
            |__ x_test_combined_embeddings.npy
            |__ x_test_embeddings.npy
            |__ x_train_combined_embeddings.npy
            |__ x_train_embeddings.npy
            |__ x_val_combined_embeddings.npy
            |__ x_val_embeddings.npy
    |__ figures
        |__ base_training_loss.png
        |__ combined_training_loss.png
    |__ predictions
        |__ base
            |__ test_nn_emotion_classification_report_base.json
            |__ test_nn_emotion_confusion_matrix_base.png
            |__ validation_nn_emotion_classification_report_base.json
            |__ validation_nn_emotion_confusion_matrix_base.png
        |__ combined
            |__ test_nn_emotion_classification_report_combined.json
            |__ test_nn_emotion_confusion_matrix_combined.png
            |__ validation_nn_emotion_classification_report_combined.json
            |__ validation_nn_emotion_confusion_matrix_combined.png
|__ output
    |__ embeddings
        |__ movie_review_embeddings.npy
    |__ figures
        |__ average_likes_per_emotion.png
        |__ emotion_counts_by_country.png
        |__ emotion_counts_by_genre.png
        |__ emotion_counts_by_release_decade.png
    |__ predicted_movie_review_emotions.csv
|__ processed
    |__ cleaned_labeled_emotion_data_supplemental.csv
    |__ cleaned_labeled_emotion_data_test_supplemental.csv
    |__ cleaned_labeled_emotion_data_test.csv
    |__ cleaned_labeled_emotion_data_train_supplemental.csv
    |__ cleaned_labeled_emotion_data_train.csv
    |__ cleaned_labeled_emotion_data_validation_supplemental.csv
    |__ cleaned_labeled_emotion_data_validation.csv
    |__ cleaned_labeled_emotion_data.csv
    |__ cleaned_movie_review_data.csv
    |__ movie_data.csv
|__ raw
    |__ all_movie_review_data.csv
    |__ movie_review_data_0_9.csv
    |__ movie_review_data_10_19.csv
    |__ movie_review_data_20_29.csv
    |__ movie_review_data_30_39.csv
    |__ movie_review_data_40_49.csv
    |__ movie_review_data_50_59.csv
    |__ movie_review_data_60_64.csv
    |__ supplemental_emotion_data.txt
    |__ twitter_emotion_corpus.txt
|__ letterboxd_top_250_urls.json
data_processing
|__ emotions
    |__ baseline_emotions.py
    |__ explore_emotions.py
    |__ process_emotions.py
    |__ split_emotions.py
|__ reviews
    |__ explore_reviews.py
    |__ merge_reviews.py
    |__ process_reviews.py
    |__ scrape_reviews.py
    |__ ublockoriginlite.crx
|__ scrape_movie_data.py
models
|__ nn_param_grid_search
    |__ all-MiniLM-L6-v2
        |__ base
            |__ metrics_lr{learning_rate}_momentum{momentum}_epochs{num_epochs}_batch{batch_size}.json
            |__ ...
        |__ combined
            |__ metrics_lr{learning_rate}_momentum{momentum}_epochs{num_epochs}_batch{batch_size}.json
            |__ ...
    |__ m3e-base
        |__ base
            |__ metrics_lr{learning_rate}_momentum{momentum}_epochs{num_epochs}_batch{batch_size}.json
            |__ ...
        |__ combined
            |__ metrics_lr{learning_rate}_momentum{momentum}_epochs{num_epochs}_batch{batch_size}.json
            |__ ...
|__ best_eec_model.pth
|__ best_model.py
|__ best_params.py
|__ emotion_embeddings.py
|__ nn.py
|__ README.MD
output
|__ inference.py
|__ visualization.py
writeups
|__ baseline
    |__ ...
|__ paper
    |__ ...
|__ proposal
    |__ ...
.gitignore
README.md
requirements.txt
utils.py
```

## Code Functionality

### Data Preprocessing

The following python files were used to scrape, preprocess, and expore the
Letterboxd movie reviews dataset:

-   `data_processing/emotions/explore_reviews.py`
-   `data_processing/emotions/merge_reviews.py`
-   `data_processing/emotions/process_reviews.py`
-   `data_processing/emotions/scrape_reviews.py`

The following python files were used to scrape, preprocess, and expore the
emotions dataset:

-   `data_processing/emotions/explore_emotions.py`
-   `data_processing/emotions/process_emotions.py`
-   `data_processing/emotions/split_emotions.py`
-   `utils.py`

### Baseline Evaluation

The following python file was used to evaluate the performance of the baseline
classification model:

-   `data_processing/emotions/baseline_emotions.py`
-   `utils.py`

### Models

The following python files were used to develop the custom classification model:

-   `models/best_model.py`
-   `models/best_params.py`
-   `models/emotion_embeddings.py`
-   `models/nn.py`
-   `utils.py`

### Emotional Analysis

The followiing python files were used to analyze the emotions expressed in the
movie reviews:

-   `data_processing/scrape_movie_data.py`
-   `output/inference.py`
-   `output/visualization.py`
-   `utils.py`
