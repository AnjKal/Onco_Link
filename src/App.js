import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LoginPage from './components/Loginpage';
import SignUpPage from './components/SignUp';
import HomePage from './HomePage';
import AIDiagnosticForm from './AIDiagnosticForm'; // Assuming this is your doctor page

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/signup" element={<SignUpPage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/diagnostic" element={<AIDiagnosticForm />} />
      </Routes>
    </Router>
  );
}

export default App;
