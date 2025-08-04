# Polygon Coloring with UNet

This project implements a UNet model in PyTorch to color polygon images based on a specified color. The model takes an image of a polygon and a color name as input and outputs an image of the polygon filled with the desired color.

## Hyperparameters

We used the following hyperparameters for training:

*   **Learning Rate:** `0.001` (Initial setting, could be tuned further)
*   **Epochs:** `50` (Maximum epochs, training stopped earlier due to early stopping)
*   **Batch Size:** `32`
*   **Loss Function:** Mean Squared Error (MSE) - Chosen to minimize the pixel-wise difference between the predicted colored image and the ground truth.
*   **Optimizer:** Adam - A popular choice for its adaptive learning rate capabilities.
*   **Early Stopping Patience:** `10` - Training stops if the validation loss does not improve for 10 consecutive epochs to prevent overfitting.

## Architecture

The core of the model is a custom implementation of the **UNet architecture**.

*   **Input:** The model takes two inputs: a polygon image (3 channels for RGB) and a color vector (6 elements for one-hot encoding of the 6 colors).
*   **Color Conditioning:** The color information is incorporated by concatenating the color vector (expanded to match the image dimensions) with the input image along the channel dimension before the first convolutional layer. This allows the network to learn to condition its output based on the desired color.
*   **Output:** The model outputs an image with 3 channels (RGB) representing the colored polygon.

No specific ablations were performed in this implementation.

## Training Dynamics

*   **Loss Curves:** The training and validation loss curves show that the loss generally decreases over epochs. The early stopping mechanism helps to prevent overfitting by stopping training when the validation loss plateaus or starts to increase.
*   **Qualitative Output Trends:** Based on the inference results, the model is able to color the polygons with the specified colors reasonably well.
*   **Typical Failure Modes:** Potential failure modes could include:
    *   Incorrect coloring for certain polygon shapes or colors, especially if they are underrepresented in the training data.
    *   Blurry or inaccurate edges of the colored polygons.
    *   Difficulty with complex or unseen polygon shapes.
*   **Fixes Attempted:** Early stopping was implemented to mitigate overfitting, which can lead to improved generalization to unseen data.

## Key Learnings

*   Implementing custom PyTorch `Dataset` and `DataLoader` is crucial for handling custom datasets and preparing data for training.
*   Incorporating additional information like color into a UNet architecture can be done by concatenating it with the input or at different layers.
*   Experiment tracking with tools like Weights & Biases is essential for monitoring training progress, comparing different runs, and identifying potential issues like overfitting.
*   Early stopping is an effective technique to prevent overfitting and improve the model's generalization performance.
*   Analyzing training and validation loss curves is vital for understanding the model's learning process and identifying overfitting.

## Inference

Inference section of this notebook demonstrates how to load the trained model and use it to color new polygon images.
