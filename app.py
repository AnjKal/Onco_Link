from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

# ----------------- SIGNUP -----------------
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    role = data.get('role')

    if role == 'doctor':
        conn = sqlite3.connect('doctors.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM doctors WHERE email = ?", (data['email'],))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': 'Doctor already exists'}), 409

        cursor.execute("""
            INSERT INTO doctors (name, age, email, password, designation)
            VALUES (?, ?, ?, ?, ?)
        """, (data['name'], data['age'], data['email'], data['password'], data['designation']))
        conn.commit()
        conn.close()
        return jsonify({'success': True})

    elif role == 'patient':
        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE email = ?", (data['email'],))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': 'Patient already exists'}), 409

        cursor.execute("""
            INSERT INTO patients (name, age, email, password, sex)
            VALUES (?, ?, ?, ?, ?)
        """, (data['name'], data['age'], data['email'], data['password'], data['sex']))
        conn.commit()
        conn.close()
        return jsonify({'success': True})

    else:
        return jsonify({'success': False, 'message': 'Invalid role'}), 400

# ----------------- LOGIN -----------------
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')

    if role == 'doctor':
        conn = sqlite3.connect('doctors.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM doctors WHERE email = ? AND password = ?", (email, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

    elif role == 'patient':
        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE email = ? AND password = ?", (email, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

    else:
        return jsonify({'success': False, 'message': 'Invalid role'}), 400
    
@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email')
    new_password = data.get('new_password')
    role = data.get('role')

    if role not in ['doctor', 'patient']:
        return jsonify({'success': False, 'message': 'Invalid role'}), 400

    db_name = 'doctors.db' if role == 'doctor' else 'patients.db'
    table_name = 'doctors' if role == 'doctor' else 'patients'

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {table_name} WHERE email = ?", (email,))
        user = cursor.fetchone()

        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        cursor.execute(f"UPDATE {table_name} SET password = ? WHERE email = ?", (new_password, email))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Password updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
