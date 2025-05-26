from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import numpy as np
from flask_cors import CORS

diagnostic_app = Flask(__name__)
CORS(diagnostic_app)  # Enable CORS for cross-origin requests

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
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

input_features = [
    'treatment_best_response', 'SLC33A1', 'NFATC4', 'SLC25A43', 
    'SLC5A10', 'SLC6A8', 'SLC29A4', 'SLC6A1', 'SLC25A40', 'SLC1A1'
]

# Load models
ft_model = FTTransformer()
ft_model.load_state_dict(torch.load(r"C:\oncolink-frontend\final_final_final_ft.pth", map_location=torch.device('cpu')))
ft_model.eval()

rf_model = joblib.load(r"C:\oncolink-frontend\final_model_rf.pkl")
scaler = joblib.load(r"C:\oncolink-frontend\scaler.pkl")

@diagnostic_app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    x_input = np.array([[data[feat] for feat in input_features]])
    x_scaled = scaler.transform(x_input)

    # FT Transformer prediction
    tensor_input = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        ft_logits = ft_model(tensor_input)
        ft_probs = torch.softmax(ft_logits, dim=1).numpy()[0]

    # Random Forest prediction
    rf_probs = rf_model.predict_proba(x_scaled)[0]

    # Ensemble
    ensemble_probs = 0.3 * ft_probs + 0.7 * rf_probs
    prediction = int(np.argmax(ensemble_probs))

    return jsonify({
        'ft_probs': ft_probs.tolist(),
        'rf_probs': rf_probs.tolist(),
        'ensemble_probs': ensemble_probs.tolist(),
        'prediction': prediction
    })

if __name__ == '__main__':
    diagnostic_app.run(port=5001, debug=True)  # changed port to 5001 to avoid conflict
