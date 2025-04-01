### **Assignment 1: Predicting Default Payments with Fully-Connected Neural Networks**

**General Description:**
This project focused on predicting credit card default payments using Fully-Connected Neural Networks (FCNNs). The objective was to build a binary classification model capable of identifying customers at risk of default based on demographic data, credit limits, payment history, and bill/payment amounts. Special attention was given to exploratory data analysis, highlighting a class imbalance (77.7% non-default vs. 22.3% default), and data preparation (scaling and encoding).

**Technologies Used:**
* **Language:** Python
* **Core Libraries:** TensorFlow/Keras (Sequential API, Dense, Batch Normalization, SGD Optimizer), Scikit-learn (train\_test\_split, StandardScaler, metrics), Pandas, NumPy, Matplotlib/Seaborn.

**Results Obtained:**
* The FCNN model achieved an accuracy of approximately **80.27%** on the validation set.
* Training and validation loss/accuracy curves showed good generalization without significant overfitting.
* Due to class imbalance, the model performed better at predicting the majority class (non-default) with an F1-score of 0.88, while struggling with the minority class (default), achieving an F1-score of **0.45** and a recall of 0.35.
* The AUC-ROC score was **0.64**, indicating moderate discriminative ability.
* The confusion matrix confirmed the tendency to correctly classify non-defaults but with difficulty in identifying defaults.

---

### **Assignment 2: Handwritten Double-Digit Classification and Autoencoders**

**General Description:**
This assignment was divided into two main parts:
1.  **Multi-class Classification:** Development and comparison of Fully-Connected (FC) models to classify grayscale images (28x39 pixels) of handwritten double digits (from 0 to 50). The impact of L2 regularization to mitigate overfitting was explored.
2.  **Autoencoder:** Construction of an Autoencoder for dimensionality reduction and image reconstruction. Reconstruction quality (using SSIM) and the ability to generate new samples from the latent space were evaluated. Finally, the use of latent (encoded) representations and PCA was compared against original data for a supervised classification task using Random Forest.

**Technologies Used:**
* **Core Libraries:** Keras (Model API for Autoencoders, L2 Regularization), Scikit-learn (Random Forest Classifier, PCA), **`skimage.metrics`** (for SSIM).

**Results Obtained:**
* **FC Classification:**
    * Non-regularized model: Test Accuracy **~93.52%**. Showed signs of overfitting.
    * L2-regularized model: Test Accuracy **~94.24%**. Demonstrated better generalization and less overfitting, making it the preferred model for robustness.
* **Autoencoder:**
    * Achieved good convergence and reconstructed images with high fidelity (Average SSIM **~0.89**).
    * Was able to generate new samples of double digits, although some appeared blurry.
    * Reconstruction sometimes improved, but often degraded subsequent classification due to loss of detail.
* **Supervised Classification Comparison:**
    * Regularized NN (original data): **94.24%**
    * Random Forest (original data): **85.28%**
    * Random Forest (Autoencoder encoded data): **77.64%**
    * Random Forest (PCA-reduced data, 90% variance): **74.42%**

---

### **Assignment 3: CNN Parameter Reduction for MNIST Classification**

**General Description:**
The goal of this assignment was to explore techniques for reducing the complexity (number of parameters) of a Convolutional Neural Network (CNN) for the MNIST handwritten digit classification task, while aiming to maintain high accuracy. Starting from a reference CNN model, two alternative models with significantly fewer parameters were developed and compared. Strategies included reducing convolutional filters, increasing depth, using different pooling methods (MaxPooling vs. AveragePooling), and introducing L1 regularization.

**Technologies Used:**
* **Core Libraries:** Keras (Layers: Conv2D, MaxPooling2D, AveragePooling2D; L1 Regularization; integrated MNIST dataset loading).

**Results Obtained:**
* **Reference Model (CNN):**
    * Parameters: **34,826**
    * Test Accuracy: **~99.10%**
* **Model 1 (Reduced Filters, Average Pooling):**
    * Parameters: **6,704** (~80% fewer than reference)
    * Test Accuracy: **~97.58%**
* **Model 2 (Deeper, Fewer Filters, L1 Reg., Intermediate Dense):**
    * Parameters: **6,776** (~80% fewer than reference)
    * Test Accuracy: **~97.38%**
* **Conclusion:** It was demonstrated that CNN parameters for MNIST can be drastically reduced while maintaining very competitive performance (over 97% accuracy) through targeted architectural changes and regularization, making the models more efficient.

---

### **Assignment 4: Character-Level Text Generation with LSTM**

**General Description:**
This project focused on character-level text generation using Recurrent Neural Networks, specifically LSTMs (Long Short-Term Memory). The source text used was Dante Alighieri's "Divina Commedia". The goal was to train a model capable of predicting the next character given a preceding sequence of characters (`maxlen`). Optimization of the `maxlen` hyperparameter was explored, and a deeper LSTM model incorporating Dropout was developed to improve performance compared to a simpler baseline model. Performance was evaluated using accuracy, loss, and perplexity.

**Technologies Used:**
* **Core Libraries:** Keras (Layers: LSTM; Optimizers: RMSprop).

**Results Obtained:**
* **Reference Model (Single LSTM layer):**
    * Parameters: **91,688**
    * Test Accuracy: **~46.74%**
    * Perplexity: **~5.53**
    * Training Time: ~31 seconds (for 20 epochs, batch 2048)
    * Showed overfitting.
* **`maxlen` Search:** The optimal value found was **15**.
* **New Model (Deeper LSTM, Dropout, maxlen=15, batch 128):**
    * Parameters: **506,408**
    * Test Accuracy: **~54.41%**
    * Perplexity: **~4.05**
    * Training Time: ~66 seconds (for 11 epochs with EarlyStopping, batch 128)
    * Showed significant improvement in accuracy and perplexity over the reference, with more contained overfitting.
* **Comparison:** The new model achieved better performance at the cost of increased complexity (more parameters) and longer training time.
