import os
import uuid
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import ASTForAudioClassification

# Set default path
path = '.'

# ===============================
# Data Processing Functions
# ===============================

def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract metadata.
    
    Example filename: 01-01-06-01-02-01-12.mp4
    """
    # Remove file extension and split by dash
    parts = os.path.splitext(filename)[0].split('-')
    
    if len(parts) != 7:
        return None
    
    modality_dict = {
        '01': 'full-AV',
        '02': 'video-only',
        '03': 'audio-only'
    }
    
    vocal_channel_dict = {
        '01': 'speech',
        '02': 'song'
    }
    
    emotion_dict = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    intensity_dict = {
        '01': 'normal',
        '02': 'strong'
    }
    
    statement_dict = {
        '01': 'Kids are talking by the door',
        '02': 'Dogs are sitting by the door'
    }
    
    actor_id = int(parts[6])
    gender = 'female' if actor_id % 2 == 0 else 'male'
    
    metadata = {
        'filename': filename,
        'modality': modality_dict.get(parts[0], parts[0]),
        'vocal_channel': vocal_channel_dict.get(parts[1], parts[1]),
        'emotion': emotion_dict.get(parts[2], parts[2]),
        'emotional_intensity': intensity_dict.get(parts[3], parts[3]),
        'statement': statement_dict.get(parts[4], parts[4]),
        'repetition': parts[5],
        'actor_id': actor_id,
        'gender': gender
    }
    
    return metadata


def create_ravdess_dataframe(directory):
    """
    Create a DataFrame containing metadata for all RAVDESS files in the directory.
    """
    metadata_list = []
    
    # Look for directories that start with "Actor"
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                metadata = parse_ravdess_filename(file)
                
                if metadata:
                    metadata['file_path'] = file_path
                    metadata_list.append(metadata)
    
    return pd.DataFrame(metadata_list)


# ===============================
# Model Definition
# ===============================

class MultiModalClassifier(nn.Module):
    """
    Multi-modal emotion classifier combining vision and audio features.
    Uses ViT for image processing and AST for audio processing.
    """
    def __init__(self, num_classes=8):
        super(MultiModalClassifier, self).__init__()
        
        # 1. Image branch: Use a pre-trained ViT model
        self.vit = models.vit_b_16(pretrained=True)
        # Replace the classification head with identity
        self.vit.heads.head = nn.Identity()
        
        # 2. Audio branch: Use AST model for audio
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", 
            attn_implementation="sdpa",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # We'll extract features before the classification layer
        self.ast_hidden_size = self.ast.config.hidden_size  # Usually 768
        
        # 3. Fusion: Concatenate features from both modalities
        self.fusion = nn.Sequential(
            nn.Linear(768 + self.ast_hidden_size, 512),  # 768 from ViT + hidden_size from AST
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        print("Model architecture:")
        print(f"  ViT output dimension: 768")
        print(f"  AST output dimension: {self.ast_hidden_size}")
        print(f"  Fusion input dimension: {768 + self.ast_hidden_size}")
    
    def forward(self, x_img, x_audio):
        # Process image through ViT
        img_features = self.vit(x_img)  # CLS token embedding
        
        # Process audio through AST
        # Make sure audio has the right shape before passing to AST
        # The AST model expects input of shape [batch_size, num_channels, height, width]
        # where height=128 (mel bins) and width is time frames
        
        # Ensure we have 4D input: [batch_size, channels, height, width]
        if x_audio.dim() > 4:
            x_audio = x_audio.squeeze(3)  # Remove any extra dimensions
        
        ast_outputs = self.ast(x_audio, output_hidden_states=True)
        
        # Extract the pooled output (CLS token representation)
        audio_features = ast_outputs.hidden_states[-1][:, 0, :]
        
        # Concatenate the features
        combined_features = torch.cat([img_features, audio_features], dim=1)
        
        # Final classification
        logits = self.fusion(combined_features)
        
        return logits


# ===============================
# Dataset Definition
# ===============================

class RAVDESSMultiModalDataset(Dataset):
    """
    Dataset for loading and preprocessing RAVDESS video data for multi-modal emotion recognition.
    Extracts both visual frames and audio spectrograms from videos.
    """
    def __init__(self, dataframe, img_size=224, spec_size=(224, 224)):
        self.dataframe = dataframe
        self.img_size = img_size
        self.spec_size = spec_size
        
        # Map emotion labels to integers
        self.emotion_to_idx = {
            'neutral': 0,
            'calm': 1,
            'happy': 2,
            'sad': 3,
            'angry': 4,
            'fearful': 5,
            'disgust': 6,
            'surprised': 7
        }
    
    def __len__(self):
        return len(self.dataframe)
    
    def extract_frame(self, video_path):
        """Extract a representative frame from the video"""
        cap = cv2.VideoCapture(video_path)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set position to middle frame
        middle_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Return a blank frame if reading fails
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Resize and convert to RGB
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        frame = frame.transpose(2, 0, 1)  # HWC to CHW
        frame = torch.from_numpy(frame).float() / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = (frame - mean) / std
        
        return frame
    
    def extract_audio_spectrogram(self, video_path):
        """Extract audio from video and convert to spectrogram for AST"""
        try:
            # Create a temporary audio file with a unique name
            temp_audio = f"temp_audio_{uuid.uuid4().hex}.wav"
            
            # Extract audio using ffmpeg
            ffmpeg_cmd = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a {temp_audio} -y"
            result = subprocess.run(ffmpeg_cmd, shell=True, stderr=subprocess.PIPE)
            
            # Check if the file was created successfully
            if not os.path.exists(temp_audio):
                print(f"Failed to extract audio from {video_path}")
                print(f"ffmpeg error: {result.stderr.decode('utf-8')}")
                # Return a dummy spectrogram
                return torch.zeros((1, 128, 1024))
            
            # Load audio
            waveform, sample_rate = torchaudio.load(temp_audio)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed (AST expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Generate log mel spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                hop_length=160,  # 10ms
                n_mels=128
            )(waveform)
            
            # Convert to decibels
            log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
            
            # Normalize
            log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.mean()) / (log_mel_spectrogram.std() + 1e-9)
            
            # Ensure consistent size
            fixed_width = 1024  # Fixed number of time frames
            if log_mel_spectrogram.shape[2] > fixed_width:
                # Truncate if too long
                log_mel_spectrogram = log_mel_spectrogram[:, :, :fixed_width]
            elif log_mel_spectrogram.shape[2] < fixed_width:
                # Pad with zeros if too short
                padding = torch.zeros((1, 128, fixed_width - log_mel_spectrogram.shape[2]))
                log_mel_spectrogram = torch.cat([log_mel_spectrogram, padding], dim=2)
            
            # Clean up
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return log_mel_spectrogram
            
        except Exception as e:
            print(f"Error processing audio for {video_path}: {str(e)}")
            # Return a dummy spectrogram
            return torch.zeros((1, 128, 1024))
            
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        video_path = row['file_path']
        emotion = row['emotion']
        
        # Extract visual and audio features
        image = self.extract_frame(video_path)
        spectrogram = self.extract_audio_spectrogram(video_path)
        
        # Get label
        label = torch.tensor(self.emotion_to_idx[emotion])
        
        return {
            'image': image,
            'audio': spectrogram,
            'label': label,
            'emotion': emotion
        }


# ===============================
# Training Functions
# ===============================

def custom_collate(batch):
    """Custom collate function to handle variable-sized tensors"""
    images = torch.stack([item['image'] for item in batch])
    
    # Get the maximum dimensions for audio spectrograms
    max_channels = max(item['audio'].shape[0] for item in batch)
    max_height = max(item['audio'].shape[1] for item in batch)
    max_width = max(item['audio'].shape[2] for item in batch)
    
    # Pad audio spectrograms to the same size
    padded_audio = []
    for item in batch:
        audio = item['audio']
        # Create a padded tensor
        padded = torch.zeros(max_channels, max_height, max_width)
        # Copy the original data
        c, h, w = audio.shape
        padded[:c, :h, :w] = audio
        padded_audio.append(padded)
    
    # Stack the padded audio tensors
    audio = torch.stack(padded_audio)
    
    # Ensure audio has the correct shape for AST: [batch_size, channels, height, width]
    # Remove the extra dimension if present
    if audio.dim() == 5:
        # Reshape from [batch, 1, 128, 1, 1024] to [batch, 1, 128, 1024]
        audio = audio.squeeze(3)
    
    # Stack labels
    labels = torch.stack([item['label'] for item in batch])
    
    # Collect emotions (strings don't need stacking)
    emotions = [item['emotion'] for item in batch]
    
    return {
        'image': images,
        'audio': audio,
        'label': labels,
        'emotion': emotions
    }


def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4, device='cuda'):
    """
    Train the multi-modal emotion recognition model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        history: Dictionary containing training and validation metrics
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images = batch['image'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, audio)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images = batch['image'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(images, audio)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_multimodal_emotion_model.pth')
    
    return history


def main():
    """Main function to run the training pipeline"""
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(ravdess_df, test_size=0.2, random_state=42, stratify=ravdess_df['emotion'])
    
    # Create datasets
    train_dataset = RAVDESSMultiModalDataset(train_df)
    val_dataset = RAVDESSMultiModalDataset(val_df)
    
    # Create data loaders with the custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=1,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=1,
        collate_fn=custom_collate
    )
    
    # Initialize model
    model = MultiModalClassifier(num_classes=8)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        lr=1e-4,
        device=device
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# ===============================
# Main Execution
# ===============================

# Create DataFrame from the dataset
print(f"Looking for RAVDESS files in: {path}")
ravdess_df = create_ravdess_dataframe(path)

# Display information about the dataset
print(f"Total files found: {len(ravdess_df)}")
if len(ravdess_df) > 0:
    print("\nEmotion distribution:")
    print(ravdess_df['emotion'].value_counts())
    
    print("\nModality distribution:")
    print(ravdess_df['modality'].value_counts())
    
    print("\nGender distribution:")
    print(ravdess_df['gender'].value_counts())
    
    # Display the first few rows
    print("\nSample data:")
    ravdess_df.head()

if __name__ == "__main__":
    main()