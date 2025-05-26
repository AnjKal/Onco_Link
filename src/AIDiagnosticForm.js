import React, { useState } from 'react';
import axios from 'axios';

function AIDiagnosticForm() {
  const featureNames = [
    'treatment_best_response', 'SLC33A1', 'NFATC4', 'SLC25A43',
    'SLC5A10', 'SLC6A8', 'SLC29A4', 'SLC6A1', 'SLC25A40', 'SLC1A1'
  ];

  const initialState = featureNames.reduce((obj, key) => {
    obj[key] = '';
    return obj;
  }, {});

  const [formData, setFormData] = useState(initialState);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formattedData = {};
    for (let key in formData) {
      formattedData[key] = parseFloat(formData[key]);
    }
    try {
      const res = await axios.post('http://localhost:5001/predict', formattedData);
      setResult(res.data);
    } catch (err) {
      console.error('Error calling API:', err);
    }
  };

  return React.createElement(
    'div',
    { className: 'p-4' },
    React.createElement('h2', {}, 'AI Diagnostic Form'),
    React.createElement(
      'form',
      { onSubmit: handleSubmit },
      ...featureNames.map((key) =>
        React.createElement(
          'div',
          { key },
          React.createElement('label', {}, key),
          React.createElement('input', {
            type: 'number',
            name: key,
            value: formData[key],
            onChange: handleChange,
            required: true,
            step: 'any'
          })
        )
      ),
      React.createElement('button', { type: 'submit' }, 'Run AI Diagnostic')
    ),
    result &&
      React.createElement(
        'div',
        { className: 'mt-4' },
        React.createElement('p', {}, `Prediction: ${result.prediction === 1 ? 'Likely to Respond' : 'Not Likely to Respond'}`),
        React.createElement('p', {}, `FT Transformer Probabilities: ${result.ft_probs.join(', ')}`),
        React.createElement('p', {}, `Random Forest Probabilities: ${result.rf_probs.join(', ')}`),
        React.createElement('p', {}, `Ensemble Probabilities: ${result.ensemble_probs.join(', ')}`)
      )
  );
}

export default AIDiagnosticForm;
