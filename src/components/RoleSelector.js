import React from 'react';

export default function RoleSelector({ role, setRole }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem', gap: '1rem' }}>
      <label>
        <input
          type="radio"
          value="doctor"
          checked={role === 'doctor'}
          onChange={(e) => setRole(e.target.value)}
        />
        Doctor
      </label>
      <label>
        <input
          type="radio"
          value="patient"
          checked={role === 'patient'}
          onChange={(e) => setRole(e.target.value)}
        />
        Patient
      </label>
    </div>
  );
}
