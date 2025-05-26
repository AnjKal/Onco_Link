import React, { useState } from 'react';
import RoleSelector from './RoleSelector';
import { useNavigate } from 'react-router-dom';
import './SignUp.css';

export default function SignUpPage() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    name: '',
    age: '',
    sex: '',
    designation: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: '',
  });

  const [error, setError] = useState('');

  const handleChange = (e) => {
    setError('');
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSignUp = async () => {
    if (formData.password !== formData.confirmPassword) {
      return setError('Passwords do not match');
    }

    const payload = {
      name: formData.name,
      age: formData.age,
      role: formData.role,
      email: formData.email,
      password: formData.password,
    };

    // Add additional fields based on role
    if (formData.role === 'doctor') {
      payload.designation = formData.designation;
    } else if (formData.role === 'patient') {
      payload.sex = formData.sex;
    }

    const response = await fetch('http://localhost:5000/api/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (result.success) {
      alert('Signed up successfully');
      navigate('/'); 
    } else {
      setError('Sign-up failed. Please try again.');
    }
  };

  return (
    <div className="signup-container">
      <div className="signup-card">
        <h2>Sign Up</h2>
        <p className="subtitle">Choose your role to begin:</p>

        <RoleSelector role={formData.role} setRole={(role) => setFormData({ ...formData, role })} />

        <input name="name" placeholder="Full Name" onChange={handleChange} />
        <input name="age" placeholder="Age" type="number" onChange={handleChange} />

        {/* Conditional fields based on role */}
        {formData.role === 'doctor' && (
          <input name="designation" placeholder="Designation" onChange={handleChange} />
        )}

        {formData.role === 'patient' && (
          <input name="sex" placeholder="Sex (e.g., Male/Female)" onChange={handleChange} />
        )}

        <input name="email" placeholder="Email" type="email" onChange={handleChange} />
        <input name="password" placeholder="Password" type="password" onChange={handleChange} />
        <input name="confirmPassword" placeholder="Confirm Password" type="password" onChange={handleChange} />

        {error && <div className="error-msg">{error}</div>}

        <button onClick={handleSignUp}>Create Account</button>
      </div>
    </div>
  );
}
