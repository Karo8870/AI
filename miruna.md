Ca pregatire pentru prima proba puteti incerca sa preziceti pretul jocurilor din setul de date de aici: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data

Ce sa aveti in vedere:
Feature engineering:
ce faceti cu date nule, diverse tipuri de inputuri etc.?
ce features sunt utile pentru antrenare?  Va poate ajuta sa vizualizati distributia datelor
exista outliers? Datele sunt echilibrate?
incercati sa nu va folositi de reviewurile in format text intr-o prima etapa (va puteti juca cu ele ulterior)

Diverse abordari:
Non-deep learning (machine learning) -- puteti folosi modele per se sau sa le combinati in diverse moduri
Deep learning -- uitati-va si pe arhitecturi de retele recurente
Transformeri


<@&1247834864871276564> 
Am mai discutat si cu comisia tehnica si revin cu concluziile discutiilor de astazi (prea multe caractere pt text pe Discord):
https://docs.google.com/document/d/1j5eCGT8_vOc_3uZghGMFisziBJAG9imqthYQ2fHGL3I/edit?usp=sharing

(Regulamentul oficial in curand!!)



In PyTorch, you can add various types of layers to a neural network. Here’s a list of some commonly used layers and a brief description of what they do:

### 1. Linear Layers
- **`nn.Linear(in_features, out_features)`**: Applies a linear transformation to the incoming data, i.e., `y = xA^T + b`.

### 2. Convolutional Layers
- **`nn.Conv1d(in_channels, out_channels, kernel_size)`**: Applies a 1D convolution over an input signal.
- **`nn.Conv2d(in_channels, out_channels, kernel_size)`**: Applies a 2D convolution over an input image.
- **`nn.Conv3d(in_channels, out_channels, kernel_size)`**: Applies a 3D convolution over a volumetric input.

### 3. Pooling Layers
- **`nn.MaxPool1d(kernel_size)`**: Applies a 1D max pooling over an input signal.
- **`nn.MaxPool2d(kernel_size)`**: Applies a 2D max pooling over an input image.
- **`nn.MaxPool3d(kernel_size)`**: Applies a 3D max pooling over a volumetric input.
- **`nn.AvgPool1d(kernel_size)`**: Applies a 1D average pooling over an input signal.
- **`nn.AvgPool2d(kernel_size)`**: Applies a 2D average pooling over an input image.
- **`nn.AvgPool3d(kernel_size)`**: Applies a 3D average pooling over a volumetric input.

### 4. Activation Layers
- **`nn.ReLU()`**: Applies the rectified linear unit function element-wise.
- **`nn.Sigmoid()`**: Applies the sigmoid function element-wise.
- **`nn.Tanh()`**: Applies the hyperbolic tangent function element-wise.
- **`nn.LeakyReLU(negative_slope)`**: Applies the leaky version of a rectified linear unit function element-wise.
- **`nn.Softmax(dim)`**: Applies the softmax function to an n-dimensional input tensor.

### 5. Dropout Layers
- **`nn.Dropout(p)`**: Randomly zeroes some of the elements of the input tensor with probability `p` using samples from a Bernoulli distribution.

### 6. Normalization Layers
- **`nn.BatchNorm1d(num_features)`**: Applies Batch Normalization over a 2D or 3D input.
- **`nn.BatchNorm2d(num_features)`**: Applies Batch Normalization over a 4D input.
- **`nn.BatchNorm3d(num_features)`**: Applies Batch Normalization over a 5D input.
- **`nn.LayerNorm(normalized_shape)`**: Applies Layer Normalization over a mini-batch of inputs.
- **`nn.InstanceNorm1d(num_features)`**: Applies Instance Normalization over a 2D or 3D input.
- **`nn.InstanceNorm2d(num_features)`**: Applies Instance Normalization over a 4D input.
- **`nn.InstanceNorm3d(num_features)`**: Applies Instance Normalization over a 5D input.

### 7. Recurrent Layers
- **`nn.RNN(input_size, hidden_size, num_layers)`**: Applies a multi-layer Elman RNN with `tanh` or `ReLU` non-linearity.
- **`nn.LSTM(input_size, hidden_size, num_layers)`**: Applies a multi-layer long short-term memory (LSTM) RNN.
- **`nn.GRU(input_size, hidden_size, num_layers)`**: Applies a multi-layer gated recurrent unit (GRU) RNN.

### 8. Embedding Layers
- **`nn.Embedding(num_embeddings, embedding_dim)`**: A simple lookup table that stores embeddings of a fixed dictionary and size.

### 9. Transformer Layers
- **`nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)`**: A transformer model.
- **`nn.TransformerEncoderLayer(d_model, nhead)`**: A single layer of the transformer encoder.
- **`nn.TransformerDecoderLayer(d_model, nhead)`**: A single layer of the transformer decoder.

### 10. Upsampling Layers
- **`nn.Upsample(scale_factor)`**: Upsamples the input using nearest neighbors.
- **`nn.ConvTranspose2d(in_channels, out_channels, kernel_size)`**: Applies a 2D transposed convolution (also known as a deconvolution).

These layers can be combined in various ways to construct complex neural network architectures tailored to specific tasks like image classification, natural language processing, or time series forecasting.




Certainly! Here’s an overview of what each category of layers is typically used for:

### 1. Linear Layers
- **Use Case**: Linear layers are fundamental components of feedforward neural networks (also known as fully connected or dense layers). They are useful for tasks where the input data is vectorized, such as in tabular data, and the output is a fixed-size vector, such as in regression or classification tasks.

### 2. Convolutional Layers
- **Use Case**: Convolutional layers are essential for processing spatial data like images and videos. They are widely used in computer vision tasks such as image classification, object detection, and segmentation, as well as in speech recognition tasks where the data can be represented as spectrograms.

### 3. Pooling Layers
- **Use Case**: Pooling layers are used to reduce the spatial dimensions (width and height) of the input, which helps in reducing the computational load and controlling overfitting. They are commonly used in conjunction with convolutional layers in computer vision tasks to achieve translation invariance and to downsample feature maps.

### 4. Activation Layers
- **Use Case**: Activation layers introduce non-linearity into the model, which is essential for learning complex patterns. Different activation functions are used depending on the problem:
  - **ReLU**: Most common in deep networks due to its simplicity and effectiveness.
  - **Sigmoid** and **Tanh**: Used in cases where outputs need to be in a specific range, such as in binary classification or in some types of recurrent networks.
  - **LeakyReLU**: Used to prevent the dying ReLU problem.
  - **Softmax**: Typically used in the final layer of a classification network to convert logits to probabilities.

### 5. Dropout Layers
- **Use Case**: Dropout layers are used to prevent overfitting by randomly setting a fraction of input units to zero at each update during training time. This helps in making the model more robust and less sensitive to specific neurons.

### 6. Normalization Layers
- **Use Case**: Normalization layers are used to stabilize and speed up the training of deep networks by normalizing the inputs of each layer. They help in making the training process more stable and efficient:
  - **BatchNorm**: Most common and used to normalize the activations of the previous layer for each mini-batch.
  - **LayerNorm**: Normalizes the inputs across the features for each sample.
  - **InstanceNorm**: Often used in style transfer and generative models, normalizes each sample independently.
  - **GroupNorm**: A compromise between batch and instance normalization, useful when batch sizes are small.

### 7. Recurrent Layers
- **Use Case**: Recurrent layers are used for sequential data, such as time series, text, or any data where the order of the data points is significant:
  - **RNN**: Good for simple sequential data, but struggles with long-term dependencies.
  - **LSTM** and **GRU**: More advanced than RNNs, capable of learning long-term dependencies, and widely used in natural language processing, speech recognition, and time series forecasting.

### 8. Embedding Layers
- **Use Case**: Embedding layers are used in natural language processing tasks to convert categorical data, such as words, into dense vectors of fixed size. These dense representations capture semantic meanings and relationships between categories.

### 9. Transformer Layers
- **Use Case**: Transformer layers are used for handling sequential data but in a parallelizable manner, making them suitable for large-scale natural language processing tasks. They excel in tasks such as language modeling, machine translation, and text generation.
  - **TransformerEncoderLayer** and **TransformerDecoderLayer**: Building blocks for the transformer architecture, useful for complex sequence-to-sequence tasks.

### 10. Upsampling Layers
- **Use Case**: Upsampling layers are used in tasks that require increasing the spatial resolution of feature maps, such as image generation, image super-resolution, and in the decoder part of autoencoders and U-Net architectures for image segmentation.
  - **ConvTranspose2d**: Used to perform learnable upsampling operations, also known as deconvolutions.

By combining these layers appropriately, you can construct powerful neural networks tailored to a wide range of tasks in computer vision, natural language processing, and other domains.