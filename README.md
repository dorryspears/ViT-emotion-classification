# Multimodal Emotion Recognition

A deep learning model for emotion recognition using both visual and audio data from the RAVDESS dataset. This project fuses features from Vision Transformers (ViT) and Audio Spectrogram Transformers (AST) to classify emotions in video recordings.

## Results

![Training Results](training_history.png)

The model was trained for 5 epochs, with the following results:
- Training loss decreased steadily from 2.0 to approximately 0.1
- Training accuracy increased to nearly 100%
- Validation accuracy reached around 52% after 4 epochs

The gap between training and validation accuracy suggests some overfitting, which could be addressed with additional regularization techniques or data augmentation.

## Dataset

This project uses the [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976) dataset, which contains:
- 24 professional actors (12 female, 12 male)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprised
- Both speech and song recordings
- Full audiovisual (AV), video-only, and audio-only versions

## Model Architecture

The model combines two powerful transformer architectures:

1. **Visual Processing**: Pre-trained Vision Transformer (ViT-B/16) extracts features from video frames
2. **Audio Processing**: Audio Spectrogram Transformer (AST) processes mel spectrograms extracted from audio
3. **Fusion Component**: A neural network that concatenates features from both modalities and performs final classification
