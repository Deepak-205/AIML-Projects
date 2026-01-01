# Music Genre Classification 🎵

## 1. Project Title
Music Genre Classification

## 2. Domain
Computer Vision (Image Classification using Convolutional Neural Networks)

## 3. Dataset
**GTZAN Music Genre Dataset**  
Source: Kaggle  

The dataset contains 1000 audio samples across 10 music genres such as blues, classical,
country, disco, hiphop, jazz, metal, pop, reggae, and rock.  
Each audio clip is 30 seconds long and stored in `.wav` format.

Dataset link:  
[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## 4. Preprocessing
- Loaded first 30 seconds of each audio file
- Converted audio signals into Mel-Spectrograms using Librosa
- Converted spectrograms to decibel scale
- Normalized values between 0 and 1
- Resized spectrograms to 128 × 128
- Applied time-frequency masking for robustness

## 5. Methodology
- Audio files were converted into Mel-Spectrogram images
- Spectrograms were used as input features
- A Convolutional Neural Network (CNN) was trained for classification
- Dense Neural Network was used as a baseline comparison model

### CNN Architecture
- Conv2D layers for feature extraction  
- MaxPooling layers for dimensionality reduction  
- Dropout layers to prevent overfitting  
- Dense layers for final classification into 10 genres  

## 6. Results
### Quantitative Results

| Metric | Dense Neural Network | CNN Model |
|------|----------------------|----------|
| Accuracy | 89.4% | 42.1% |
| Precision | 0.894 | 0.47 |
| Recall | 0.894 | 0.42 |
| F1 Score | 0.893 | 0.41 |

Visual results such as confusion matrix and loss curves are included in the notebook.

## 7. Discussion
The CNN model performed reasonably well on most genres but showed confusion
between similar-sounding genres such as rock and metal.  
While deeper models provided better feature learning, they also increased training
time and overfitting. The architecture was kept simple to ensure faster training
on Google Colab.

## 8. Proposed Method
A balanced deep learning approach combining CNN and Dense layers was proposed.
The system converts audio signals into Mel-Spectrogram images and performs
classification using deep learning while maintaining low computational cost.

## 9. Status
Completed ✅

## 10. Notebook
The complete implementation and results are available in:  
`Music_Genre_Classification.ipynb`
