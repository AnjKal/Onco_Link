from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

users = []  # Temporary in-memory "database"

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')

    if any(u['email'] == email for u in users):
        return jsonify({'success': False, 'message': 'User already exists'}), 409

    user = {
        'name': data.get('name'),
        'age': data.get('age'),
        'role': data.get('role'),
        'email': email,
        'password': data.get('password'),
        'designation': data.get('designation') if data.get('role') == 'doctor' else '',
        'sex': data.get('sex') if data.get('role') == 'patient' else '',
    }

    users.append(user)
    print(f" New user registered: {user}")
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
