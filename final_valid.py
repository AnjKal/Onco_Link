import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#FT transformer
class FTTransformer(nn.Module):
    def __init__(self, input_dim=10, num_classes=2, d_model=128, nhead=4, num_layers=1, dropout=0.3):
        super(FTTransformer, self).__init__()
        self.feature_tokenizer = nn.Linear(input_dim, d_model)
        self.dropout1 = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.feature_tokenizer(x)
        x = self.dropout1(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)   # Mean pooling
        return self.classifier(x)

# Load FT-Transformer onto CPU
ft_model = FTTransformer(input_dim=10, num_classes=2, d_model=128, nhead=4, num_layers=1, dropout=0.3)
state_dict = torch.load(
    "D:\\EL_sem4\\my_code\\final_final_final_ft.pth",
    map_location=torch.device('cpu')
)
ft_model.load_state_dict(state_dict)
ft_model.eval()

# Load RF model
rf_model = joblib.load("D:\\EL_sem4\\final_model_rf.pkl")

#Single user input

# Input dictionaries
user_input = {
    'treatment_best_response' : 1,
    'SLC33A1' :4.621 , 
    'NFATC4' : 4.5551, 
    'SLC25A43' :3.8754 , 
    'SLC5A10' : 0.6132 , 
    'SLC6A8' : 5.5892, 
    'SLC29A4' :5.4186 , 
    'SLC6A1' :0.7502 ,
    'SLC25A40' : 3.2017,
    'SLC1A1' : 4.411
}


# Feature order
input_features = [
    'treatment_best_response', 'SLC33A1', 'NFATC4', 'SLC25A43', 
    'SLC5A10', 'SLC6A8', 'SLC29A4', 'SLC6A1','SLC25A40','SLC1A1'
]




# Prepare input for FT Transformer
input_np = np.array([[user_input[feat] for feat in input_features]])
input_tensor_ft = torch.tensor(input_np, dtype=torch.float32)



# FT Transformer prediction
with torch.no_grad():
    ft_logits = ft_model(input_tensor_ft)
    ft_probs = torch.softmax(ft_logits, dim=1).numpy()[0]

# Random Forest prediction
input_df_rf = pd.DataFrame(input_np, columns=input_features)
rf_probs = rf_model.predict_proba(input_df_rf)[0]

# Ensemble (soft voting)
ensemble_probs = (0.3*ft_probs + 0.7*rf_probs) 
ensemble_prediction = np.argmax(ensemble_probs)

# Output
print("FT Transformer probs:", ft_probs)
print("Random Forest probs:", rf_probs)
print("Ensembled probs:", ensemble_probs)
print("Final Prediction (0 or 1):", ensemble_prediction)

#Full dataset

# Load the full dataset
'''df = pd.read_csv('D:\\EL_sem4\\my_code\\final_data.csv')

input_features = [
    'treatment_best_response', 'SLC33A1', 'NFATC4', 'SLC25A43', 
    'SLC5A10', 'SLC6A8', 'SLC29A4', 'SLC6A1','SLC25A40','SLC1A1'
]

# Load or fit scaler
scaler_path = 'D:\\EL_sem4\\my_code\\scaler.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = StandardScaler()
    scaler.fit(df[input_features])
    joblib.dump(scaler, scaler_path)

# Ensure the features are in the correct order
X = df[input_features].values
y_true = df['response'].values  # Replace 'response' with your actual label column name

# FT Transformer predictions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Ideally use .transform() with saved scaler

input_tensor_ft = torch.tensor(X_scaled, dtype=torch.float32)

with torch.no_grad():
    ft_logits = ft_model(input_tensor_ft)
    ft_probs = torch.softmax(ft_logits, dim=1).numpy()

# Random Forest predictions
rf_probs = rf_model.predict_proba(df[input_features])

# Ensemble (soft voting)
ensemble_probs = ((0.3*ft_probs) + (0.7*rf_probs))
ensemble_predictions = np.argmax(ensemble_probs, axis=1)

# Output metrics


print("FT Transformer Accuracy:", accuracy_score(y_true, np.argmax(ft_probs, axis=1)))
print("Random Forest Accuracy:", accuracy_score(y_true, np.argmax(rf_probs, axis=1)))
print("Ensemble Accuracy:", accuracy_score(y_true, ensemble_predictions))
print("\nClassification Report (Ensemble):\n", classification_report(y_true, ensemble_predictions))
print("\nConfusion Matrix (Ensemble):\n", confusion_matrix(y_true, ensemble_predictions))'''