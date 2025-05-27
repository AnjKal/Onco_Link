import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Loginpage.css';

export default function LoginPage() {
  const [role, setRole] = useState('patient');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async () => {
    if (!email || !password) {
      setError('Please enter email and password');
      return;
    }

    const payload = { email, password, role };

    try {
      const response = await fetch('http://localhost:5000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (result.success) {
        alert('Login successful!');
        if (role === 'doctor') {
          navigate('/diagnostic');
        } else {
          navigate('/home');
        }
      } else {
        setError(result.message || 'Login failed. Please try again.');
      }
    } catch (err) {
      setError('Error connecting to server');
    }
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h1>Welcome to Oncolink!</h1>
        <p className="sub">Log in to your account</p>

        <div className="role-selector">
          <label>
            <input
              type="radio"
              name="role"
              value="doctor"
              checked={role === 'doctor'}
              onChange={() => setRole('doctor')}
            />
            Doctor
          </label>
          <label>
            <input
              type="radio"
              name="role"
              value="patient"
              checked={role === 'patient'}
              onChange={() => setRole('patient')}
            />
            Patient
          </label>
        </div>

        <label>Email</label>
        <input
          type="text"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <label>Password</label>
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <div className="options">
          <label>
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={() => setRememberMe(!rememberMe)}
            />
            Remember me
          </label>
          <span className="reset" onClick={() => navigate('/reset-password')}>
          Reset Password?
          </span>

        </div>

        {error && <div className="error-msg">{error}</div>}

        <button className="login-button" onClick={handleLogin}>
          Sign In
        </button>

        <p className="register">
          Don't have an account yet? <a href="/signup">Register here</a>
        </p>
      </div>
    </div>
  );
}
