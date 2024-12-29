## Project Overview
what the project does:
- Load and preprocess image datasets.
- Convert images into the Lab color space.
- Use a deep learning model with an encoder-decoder architecture for colorization.
- Train the model to predict the `AB` color channels from the `L` (lightness) channel.
- Save and visualize the trained model's results.


---

## Code Explanation in `ModelTraining.ipynb`

### 1. **Dataset Preprocessing**

#### Loading Image Data
The `load_image_dataset` function loads all images from a given folder path, applies preprocessing transformations, and converts them to tensors.
- **Input**: Path to image folder.
- **Output**: A PyTorch tensor containing all preprocessed images.

#### Paths for Training and Validation Data
- `train_folder` and `val_folder` point to the respective directories in the ImageNet dataset.

#### Preprocessing Transformations
- **Resize**: Images are resized to 128x128 pixels.
- **ToTensor**: Images are converted into tensors scaled to a `[0, 1]` range.

#### Data Loaders
The training and validation datasets are wrapped into PyTorch DataLoader objects, enabling efficient batch processing.

---

### 2. **Lab Color Space Conversion**

The Lab color space separates lightness (`L`) and color (`AB`) channels. The `preprocess_lab` function converts RGB images to Lab and extracts:
- `X`: The `L` (lightness) channel as input.
- `Y`: The `AB` color channels as target labels, scaled to [-1, 1].

---

### 3. **Model Architecture**

The colorization model consists of:

> [!Important] Architecture
> #### Encoder (VGG-16)
> - A pre-trained VGG-16 model is used as the encoder.
> - Only the first 23 layers (up to the last convolutional layer) are retained.
> - The encoder's weights are frozen to prevent updates during training.
> 
> #### Decoder (Vanilla CNN)
> - A custom CNN is used as the decoder, which progressively upsamples the encoded feature maps to the original image size (128x128).
> - The decoder uses:
>   - Convolutional layers with ReLU activation.
>   - Upsampling layers using bilinear interpolation.
>   - A final Tanh activation layer to scale the output.
> 
#### Combined Model
The `ColorizationModel` class integrates the encoder and decoder into a single model.

---

### 4. **Training Loop**

The training loop optimizes the model using:
- **Loss Function**: Mean Squared Error (MSE) loss between predicted and true `AB` channels.
- **Optimizer**: Adam optimizer with a learning rate of 0.0002.

#### Key Steps:
1. Load a batch of grayscale images and preprocess them.
2. Repeat the `L` channel to create a 3-channel input for the encoder.
3. Forward pass through the model.
4. Compute the loss and backpropagate to update the decoder weights.
5. Print the average loss after each epoch.

---

### 5. **Model Saving**

The trained model's parameters are saved to a file (`colorization_model0.pt`) for future use.
```python
torch.save(model.state_dict(), "colorization_model0.pt")
```

---

### 6. **Testing and Visualization**

#### Testing Function
The `test_and_visualize` function evaluates the model on a sample image and visualizes the colorization results.

Steps:
1. Load the model and set it to evaluation mode.
2. Convert the image to the Lab color space and extract the `L` channel.
3. Forward pass through the model to predict `AB` channels.
4. Combine the `L` and predicted `AB` channels to reconstruct the color image.

#### Visualization
- The original and colorized images are displayed side-by-side using Matplotlib.

---

## Dataset
The ImageNet dataset (7GB) is used for training and validation. Images are organized into separate folders for training and validation data.

---

## Environment
The code is designed to run on Kaggle, which provides 30 hours of free GPU usage per week. It leverages CUDA if available for faster training and inference.

---

## File Structure
- **Main Script**: Contains all preprocessing, model definition, training, and testing code.
- **Saved Model**: The trained model parameters (`colorization_model0.pt`).
- **Dataset Folders**: Organized into `train` and `val` directories.

---

> [!faq] Requirements
> 
> - Python 3.x
> - PyTorch
> - torchvision
> - scikit-image
> - numpy
> - matplotlib

---

> [!example] INSTRUCTIONS
> 
> 
> 1. **Set Up the Environment**
>    Ensure all dependencies are installed. Use Kaggle or a local machine with GPU support.
> 
> 2. **Dataset Preparation**
>    Download and extract the ImageNet dataset. Place it in the appropriate folder structure:
>    ```
>    /imagenet
>       /train
>       /val
>    ```
> 
> 3. **Run the Code**
>    Execute the script in an environment with GPU support for faster training. Use Kaggle for free GPU hours if needed.
> 
> 4. **Testing**
>    Use the `test_and_visualize` function to evaluate the model on a sample image.
> 

---


 ## Use our GUI
 - we have a gui file called `gui.py` in the repository, download it along with the `model.pt` file in the same directory
 - type in the terminal `streamlit run [absolute path to the gui.py file here]`
 - or use our web app [here](https://grayimgcolorizer.streamlit.app/) !

---

## Model 
> -[Pytorch File](https://drive.google.com/file/d/165qiDl-OMgpDFZn9u6DY69eEaMv_JthT/view?usp=sharing)
> -[Test Data](https://drive.google.com/drive/folders/1DwsO-znt3v0sJjwmjKoSAb61wr9ou_rB?usp=drive_link)
