# DCGan-using-Tensorflow
## Overview

This repository contains a Jupyter notebook implementing a DCGAN using TensorFlow and Keras for image generation. The DCGAN consists of a generator and a discriminator trained adversarially.

## Dependencies

Make sure you have the following dependencies installed in your notebook environment:

```python
!pip install tensorflow numpy matplotlib
``` 
## Dataset
The code uses the CelebA dataset, which can be downloaded and unzipped using the following commands:  
  !pip install -q kaggle  
  !mkdir ~/.kaggle  
  !cp kaggle.json ~/.kaggle/  
  !chmod 600 /root/.kaggle/kaggle.json  
  !kaggle datasets download -d jessicali9530/celeba-dataset  
  !unzip "/content/celeba-dataset.zip" -d "/content/dataset/"  
  
## Model Architecture
### Generator
The generator is defined as follows:  
  -Input: Latent vector of size LATENT_DIM   
  -Dense layer: Output size 88LATENT_DIM  
  -Reshape layer: Reshapes output to (8, 8, LATENT_DIM)  
  -Several Conv2DTranspose layers with BatchNormalization and LeakyReLU activation  
  -Output Conv2DTranspose layer with tanh activation  
### Discriminator  

The discriminator is defined as follows:  
  -Input: Image with shape (128, 128, 3)  
  -Several Conv2D layers with BatchNormalization and LeakyReLU activation  
  -Flatten layer  
  -Dense layer with sigmoid activation  
  
## GAN
The GAN class wraps the discriminator and generator and includes training logic.  

## Training
The model is trained using the CelebA dataset with a custom GAN training loop. The training process involves optimizing the discriminator and generator iteratively.  

## Callback
The ShowImage callback generates sample images from the generator during training and saves them to the "generated" folder.  

## Checkpoint
A ModelCheckpoint callback is set up to save the weights of the generator when the generator loss is minimized during training. The checkpoints are saved to the specified file path.  

## Usage
Ensure you have the necessary dependencies installed and the CelebA dataset downloaded. Run the code cells in the notebook and monitor the training process. Sample images will be generated and saved in the "generated" folder, and the generator's weights will be saved to the specified checkpoint file.




