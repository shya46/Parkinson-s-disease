import os
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import librosa
import scipy.io.wavfile as wav
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

def get_next_filename():
    """Generate a unique filename inside a dedicated folder ('recordings')."""
    folder = "recordings"
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    i = 1
    while os.path.exists(os.path.join(folder, f"test_audio{i}.wav")):
        i += 1
    return os.path.join(folder, f"test_audio{i}.wav")

def record_audio(filename, duration=5, fs=22050):
    """Record audio for a given duration and save as WAV."""
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    wav.write(filename, fs, (audio * 32767).astype(np.int16))
    print("✅ Recording saved as", filename)

def extract_features(filename):
    y, sr = sf.read(filename)
    
   
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    
    jitter = np.std(np.diff(y)) / (np.mean(y) + 1e-6)  # Avoid divide by zero
    shimmer = np.std(y) / (np.mean(y) + 1e-6)

    
    hnr = librosa.effects.harmonic(y).mean()
    
    
    features = np.hstack([mfccs_mean, jitter, shimmer, hnr])
    
    print(f"✅ Features Extracted: {features.shape}")  # Should be (16,)
    return features


csv_filename = "features.csv"


if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)
else:
    df = pd.DataFrame(columns=[f"MFCC_{i}" for i in range(13)] + ["Jitter", "Shimmer", "HNR", "status"])
    df.to_csv(csv_filename, index=False)

audio_filename = get_next_filename()
record_audio(audio_filename)  

print("\n🔍 Extracting Features from Recorded Audio...")  
live_features = extract_features(audio_filename)

print(f"✅ Features Extracted: {len(live_features)}")  
print(f"📂 CSV Columns Count: {len(df.columns)}")  

new_data = pd.DataFrame([list(live_features) + [0]], columns=df.columns)

df = pd.concat([df, new_data], ignore_index=True)
df.to_csv(csv_filename, index=False)

print(f"✅ Features saved to {csv_filename}")


if 'name' in df.columns:
    df.drop(columns=['name'], inplace=True)

print("Missing Values:\n", df.isnull().sum())

if 'filename' in df.columns:
    X = df.drop(columns=['status', 'filename'])
else:
    X = df.drop(columns=['status'])

y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Dataset Shape: {X_train.shape}")

model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ Model Accuracy: {accuracy:.4f}")
print(f"📉 Mean Absolute Error (MAE): {mae:.4f}")
print(f"📊 Root Mean Squared Error (RMSE): {rmse:.4f}")


feature_importance = pd.Series(model.feature_importances_, index=X.columns)
important_features = feature_importance.nlargest(5).index.tolist()  # Top 5 features
print("🔹 Top 5 Most Important Features:", important_features)

feature_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.show(block=False)
