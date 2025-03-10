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
from transformers import ASTModel, AutoFeatureExtractor

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
        
        # 2. Audio branch: Use AST model for audio without device mapping
        self.ast_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        
        # Load model with simplified settings - no half precision
        try:
            self.ast = ASTModel.from_pretrained(self.ast_model_name)
            
            # Add feature extractor for proper preprocessing
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.ast_model_name)
            
            # Get the hidden size from the AST model
            self.ast_hidden_size = self.ast.config.hidden_size  # Should be 768
        except Exception as e:
            print(f"Error loading AST model: {e}")
            # Fallback to a simpler model if AST fails to load
            self.ast = None
            self.ast_hidden_size = 768
            print("Using fallback audio processing")
        
        # 3. Fusion: Concatenate features from both modalities
        self.fusion = nn.Sequential(
            nn.Linear(768 + self.ast_hidden_size, 512),
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
        
        # Process audio through AST with better error handling
        try:
            batch_size = x_audio.size(0)
            
            # Handle 5D tensor case
            if x_audio.dim() == 5:
                x_audio = x_audio.squeeze(3)  # Remove the extra dimension
            
            # Check if tensor is valid
            if x_audio.numel() == 0 or torch.isnan(x_audio).any() or torch.isinf(x_audio).any():
                raise ValueError("Invalid audio tensor with zeros, NaNs or Infs")
            
            # Memory optimization: process in smaller chunks if batch is large
            if batch_size > 2 and self.ast is not None:
                # Process audio in smaller batches to save memory
                audio_features_list = []
                sub_batch_size = 2  # Process 2 samples at a time
                
                for i in range(0, batch_size, sub_batch_size):
                    # Get sub-batch
                    end_idx = min(i + sub_batch_size, batch_size)
                    x_audio_sub = x_audio[i:end_idx]
                    
                    # Reshape from [batch_size, 1, 128, 1024] to [batch_size, 1024, 128]
                    # Keep the same float32 dtype
                    x_audio_sub = x_audio_sub.squeeze(1).transpose(1, 2)
                    
                    # Clear CUDA cache to free up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process audio
                    ast_outputs = self.ast(
                        input_values=x_audio_sub,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    sub_audio_features = ast_outputs.pooler_output
                    
                    audio_features_list.append(sub_audio_features)
                
                # Concatenate all sub-batches
                audio_features = torch.cat(audio_features_list, dim=0)
                
            elif self.ast is not None:
                # For small batches, process normally
                x_audio = x_audio.squeeze(1).transpose(1, 2)
                
                # Process audio
                ast_outputs = self.ast(
                    input_values=x_audio,
                    output_hidden_states=True,
                    return_dict=True
                )
                audio_features = ast_outputs.pooler_output
            else:
                # Fallback if AST is not available
                audio_features = torch.randn(batch_size, self.ast_hidden_size, device=x_img.device)
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            # Fallback - use random features with same dtype as image features
            audio_features = torch.randn(x_img.size(0), self.ast_hidden_size, 
                                        dtype=x_img.dtype, 
                                        device=x_img.device)
        
        # Ensure consistent dtype before concatenation
        if audio_features.dtype != img_features.dtype:
            audio_features = audio_features.to(img_features.dtype)
        
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
            # Create a unique temporary file for audio extraction
            temp_audio_file = f"temp_audio_{uuid.uuid4()}.wav"
            
            # Extract audio from video using ffmpeg
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path, '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', temp_audio_file
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            
            # Load audio file
            waveform, sample_rate = torchaudio.load(temp_audio_file)
            
            # Delete temporary file
            os.remove(temp_audio_file)
            
            # Resample if needed
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                sample_rate = 16000
            
            # Handle empty or short waveforms
            if waveform.numel() == 0 or waveform.shape[1] < sample_rate * 0.5:  # Less than 0.5 seconds
                print(f"Warning: Very short or empty audio in {video_path}")
                return torch.zeros((1, 128, 1024), dtype=torch.float32)
            
            # Convert to mel spectrogram
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=128
            )(waveform)
            
            # Convert to decibels
            mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            
            # Normalize to match AST expectations
            mel_spec_db_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-10)
            
            # Ensure consistent shape for AST
            # Target shape: [1, 128, 1024]
            # Pad or trim if needed
            target_length = 1024
            current_length = mel_spec_db_norm.shape[2]
            
            if current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                log_mel_spectrogram = torch.nn.functional.pad(mel_spec_db_norm, (0, padding))
            else:
                # Trim to target length
                log_mel_spectrogram = mel_spec_db_norm[:, :, :target_length]
            
            # Make sure we have the right dtype
            log_mel_spectrogram = log_mel_spectrogram.to(torch.float32)
            
            return log_mel_spectrogram
            
        except Exception as e:
            print(f"Error processing audio for {video_path}: {str(e)}")
            # Return a dummy spectrogram with correct shape and dtype
            return torch.zeros((1, 128, 1024), dtype=torch.float32)
            
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        video_path = row['file_path']
        emotion = row['emotion']
        
        # Extract visual and audio features
        image = self.extract_frame(video_path)
        spectrogram = self.extract_audio_spectrogram(video_path)
        
        # Ensure spectrogram has the correct shape [channels, height, width]
        if spectrogram.dim() > 3:
            # Reshape to exactly [channels, height, width]
            channels, height = 1, 128
            try:
                spectrogram = spectrogram.reshape(channels, height, 1024)
            except RuntimeError:
                # Create a new tensor
                spectrogram = torch.zeros(1, 128, 1024, dtype=torch.float32)
        
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
    
    # Process audio tensors to ensure correct shape before stacking
    processed_audio = []
    for item in batch:
        audio = item['audio']
        # Ensure audio has exactly shape [1, 128, 1024]
        if audio.dim() != 3 or audio.shape[0] != 1 or audio.shape[1] != 128 or audio.shape[2] != 1024:
            # Force reshape to correct dimensions
            try:
                audio = audio.reshape(1, 128, 1024)
            except RuntimeError:
                # If reshape fails, create a new tensor with zeros
                audio = torch.zeros(1, 128, 1024, dtype=torch.float32)
        processed_audio.append(audio)
    
    # Stack audio tensors
    audio = torch.stack(processed_audio)
    
    # Stack labels
    labels = torch.stack([item['label'] for item in batch])
    
    # Collect emotions
    emotions = [item['emotion'] for item in batch]
    
    return {
        'image': images,
        'audio': audio,
        'label': labels,
        'emotion': emotions
    }


def train_model(model, train_loader, val_loader, num_epochs=5, lr=3e-5, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use a lower learning rate for AST parts of the model
    # Group parameters for optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
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
            
            # Fix audio shape if needed
            if audio.dim() == 5 and audio.shape[1] == 1 and audio.shape[3] == 1:
                audio = audio.squeeze(3)
            
            labels = batch['label'].to(device)
            
            # Zero the parameter gradients
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
                
                # Fix audio shape if needed
                if audio.dim() == 5 and audio.shape[1] == 1 and audio.shape[3] == 1:
                    audio = audio.squeeze(3)
                
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
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
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
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
    
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
    
    # Use smaller batch size to avoid memory issues
    batch_size = 4  # Reduced from 8
    
    # Create data loaders with the custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2,
        collate_fn=custom_collate
    )
    
    # Initialize model
    model = MultiModalClassifier(num_classes=8)
    
    # Train model with optimized hyperparameters
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        lr=3e-5,
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
    plt.savefig('training_history.png')
    plt.show()


# ===============================
# Main Execution
# ===============================

def load_or_create_dataframe(path, cache_file='ravdess_df.pkl'):
    """Load DataFrame from cache or create it if cache doesn't exist"""
    if os.path.exists(cache_file):
        print(f"Loading cached DataFrame from {cache_file}")
        return pd.read_pickle(cache_file)
    else:
        print(f"Creating new DataFrame from files in {path}")
        df = create_ravdess_dataframe(path)
        # Save to cache
        df.to_pickle(cache_file)
        return df

# Create DataFrame from the dataset
ravdess_df = load_or_create_dataframe(path)

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
    print(ravdess_df.head())

if __name__ == "__main__":
    main()