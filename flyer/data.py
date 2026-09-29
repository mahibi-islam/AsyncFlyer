from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ALIASES = {
    'nitrogen': ['N', 'n', 'Nitrogen'], 'phosphorus': ['P', 'p', 'Phosphorus'],
    'potassium': ['K', 'k', 'Potassium'], 'ph': ['ph', 'pH', 'PH'],
    'rainfall': ['rainfall', 'Rainfall'], 'temperature': ['temperature', 'Temperature', 'temp'],
    'humidity': ['humidity', 'Humidity'], 'moisture': ['moisture', 'Moisture', 'soil moisture'],
    'soil_color': ['soil_color', 'Soil colour', 'Soil color', 'soil colour'],
    'label': ['label', 'crop', 'Crop', 'crop_type', 'Crop Type', 'Recommended Crop']
}

def _find(df, names):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower: return lower[n.lower()]
    return None

def load_federated_data(path: str, seed: int = 42, test_size: float = .2, cloud_holdout_size: float = .2, num_clients: int = 4):
    df = pd.read_csv(Path(path))
    target = _find(df, ALIASES['label'])
    if target is None: raise ValueError(f'Could not find crop target column. Columns: {list(df.columns)}')
    numeric = []
    for key in ['nitrogen','phosphorus','potassium','ph','rainfall','temperature','humidity','moisture']:
        col = _find(df, ALIASES[key])
        if col is not None: numeric.append(col)
    if len(numeric) < 3: raise ValueError('Need at least three soil/environment numeric columns.')
    X = df[numeric].apply(pd.to_numeric, errors='coerce').fillna(df[numeric].apply(pd.to_numeric, errors='coerce').median()).to_numpy(dtype='float32')
    y_raw = df[target].astype(str).str.strip().to_numpy()
    enc = LabelEncoder(); y = enc.fit_transform(y_raw).astype('int64')
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    client_pool, cloud_idx = train_test_split(train_idx, test_size=cloud_holdout_size, random_state=seed, stratify=y[train_idx])
    scaler = StandardScaler().fit(X[client_pool]); X = scaler.transform(X).astype('float32')
    rng = np.random.default_rng(seed); shuffled = rng.permutation(client_pool)
    client_indices = [a for a in np.array_split(shuffled, num_clients)]
    def seq(indices): return X[indices][:, None, :], y[indices]
    return {'clients': [seq(i) for i in client_indices], 'cloud': seq(cloud_idx), 'test': seq(test_idx), 'classes': enc.classes_.tolist(), 'feature_names': numeric, 'scaler': scaler}
