# ResNet18-Drawing-Digits-Recognition-MNIST
This project explores training a ResNet18 model from scratch for digit classification, combined with a user interface for drawing digits. Users can interact with the program by drawing digits on a canvas, and the trained model predicts the most likely corresponding digit based on the drawn image.

## Functionality

Drawing Canvas: A user interface is created using Pygame that provides a drawing canvas where users can sketch digits with their mouse.
Digit Preprocessing: When the user finishes drawing, the drawn pixels are converted from a Pygame grid format to a format suitable for the model (e.g., a NumPy array or a PyTorch tensor).
Model Training: The ResNet18 model defined in cnn.py is trained from scratch using the MNIST dataset for digit classification.
Model Evaluation and Prediction: After training, the model is evaluated on a separate test set from the MNIST dataset. It also predicts digits based on user drawings using the processed image data.
Prediction Visualization: The predicted digit is displayed alongside the drawn image, providing feedback to the user.


![תמונה1](https://github.com/nick860/ResNet18-Drawing-Digits-Recognition--MNIST-/assets/55057278/ea3982ab-c3db-431d-8030-ccbbb3f3fb80)
