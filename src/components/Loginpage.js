import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Loginpage.css';

export default function LoginPage() {
  const [role, setRole] = useState('patient');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const navigate = useNavigate();

  const handleLogin = () => {
    // Mock authentication: just check if fields are filled
    if (email && password) {
      alert('Login successful!');
      if (role === 'doctor') {
        navigate('/diagnostic');
      } else {
        navigate('/home');
      }
    } else {
      alert('Login failed: Please enter email and password.');
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

        <label>Username</label>
        <input
          type="text"
          placeholder="Email or Phone Number"
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
          <span className="reset">Reset Password?</span>
        </div>

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