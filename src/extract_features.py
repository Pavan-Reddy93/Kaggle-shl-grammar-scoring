import librosa
import numpy as np

def extract_acoustic_features(file_path):
    """
    Extracts a robust set of 35 acoustic features from a given .wav file.
    Features: Duration, Spectral Centroid, Spectral Rolloff, RMS Energy,
             Zero Crossing Rate, and 13-dimensional MFCCs (Means and Stds).
    """
    try:
        # Load audio with fixed 16kHz sampling rate
        y, sr = librosa.load(file_path, sr=16000)

        # Duration
        duration = librosa.get_duration(y=y, sr=sr)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Spectral Centroid
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(spec_centroid)
        centroid_std = np.std(spec_centroid)

        # Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(spec_rolloff)
        rolloff_std = np.std(spec_rolloff)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # Feature dictionary mapping
        features = {
            'duration': duration,
            'centroid_mean': centroid_mean,
            'centroid_std': centroid_std,
            'rolloff_mean': rolloff_mean,
            'rolloff_std': rolloff_std,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std
        }
        for i in range(13):
            features[f'mfcc_mean_{i}'] = mfcc_mean[i]
            features[f'mfcc_std_{i}'] = mfcc_std[i]

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None