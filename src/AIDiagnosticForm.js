import React, { useState } from 'react';
import './AIDiagnosticForm.css'; // optional if you want styling

const AIDiagnosticForm = () => {
  const [inputs, setInputs] = useState({
    treatment_best_response: '',
    SLC33A1: '',
    NFATC4: '',
    SLC25A43: '',
    SLC5A10: '',
    SLC6A8: '',
    SLC29A4: '',
    SLC6A1: '',
    SLC25A40: '',
    SLC1A1: ''
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setInputs({
      ...inputs,
      [e.target.name]: parseFloat(e.target.value)
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const response = await fetch('http://localhost:5051/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputs)
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Server error');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err.message);
    }
  };

  return (
    <div className="form-container">
      <h2>AI Diagnostic Predictor</h2>
      <form onSubmit={handleSubmit} className="form-grid">
        {Object.keys(inputs).map((key) => (
          <div key={key} className="form-field">
            <label>{key}</label>
            <input
              type="number"
              step="any"
              name={key}
              value={inputs[key]}
              onChange={handleChange}
              required
            />
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>

      {error && <div className="error">Error: {error}</div>}

      {result && (
        <div className="result">
          <h3>Prediction: <span>{result.label}</span></h3>
          <p><strong>FT Transformer Probabilities:</strong> {result.ft_probs.map(p => p.toFixed(3)).join(', ')}</p>
          <p><strong>Random Forest Probabilities:</strong> {result.rf_probs.map(p => p.toFixed(3)).join(', ')}</p>
          <p><strong>Ensembled Probabilities:</strong> {result.ensemble_probs.map(p => p.toFixed(3)).join(', ')}</p>
        </div>
      )}
    </div>
  );
};

export default AIDiagnosticForm;
