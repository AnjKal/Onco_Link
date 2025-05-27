import React, { useState } from 'react';
import './Loginpage.css';

export default function ResetPasswordPage() {
  const [email, setEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [role, setRole] = useState('patient');
  const [message, setMessage] = useState('');

  const handleReset = async () => {
    if (!email || !newPassword) {
      setMessage('Please fill all fields');
      return;
    }

    const payload = { email, new_password: newPassword, role };

    try {
      const response = await fetch('http://localhost:5000/api/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      if (result.success) {
        setMessage('Password reset successful!');
      } else {
        setMessage(result.message || 'Failed to reset password');
      }
    } catch (err) {
      setMessage('Server error. Try again later.');
    }
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h1>Reset Password</h1>

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
          placeholder="Enter your email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <label>New Password</label>
        <input
          type="password"
          placeholder="Enter new password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
        />

        <button className="login-button" onClick={handleReset}>
          Reset Password
        </button>

        {message && <div className="error-msg">{message}</div>}
      </div>
    </div>
  );
}
