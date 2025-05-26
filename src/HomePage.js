import React from 'react';
import './styles.css';

function HomePage() {
  return (
    <div className="homepage">
      <h1 className="title">Oncolink</h1>
      <div className="overlay">
        <div className="description-box">
          <p>
            Oncolink is designed to empower cancer patients by combining cutting-edge science and compassionate care in one easy-to-use platform...
          </p>
          <p>You’re never alone on your journey with Oncolink by your side.</p>
        </div>
      </div>
      <div className="button-container-horizontal">
        <button className="nav-button" onClick={() => window.location.href = "/diagnostic"}>AI Diagnostic</button>
        <button className="nav-button" onClick={() => window.location.href = "https://lungcancer-cn-4.onrender.com"}>Communications</button>
        <button className="nav-button" onClick={() => window.location.href = "/meal-planner/planner.html"}>Meal Planner</button>
      </div>
    </div>
  );
}

export default HomePage;
