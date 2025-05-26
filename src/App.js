import React from 'react';
import './styles.css';

function App() {
  return (
    <div className="homepage">
      <h1 className="title">Oncolink</h1>
      
      <div className="overlay">
        <div className="description-box">
          <p>
            Oncolink is designed to empower cancer patients by combining cutting-edge science and compassionate care in one easy-to-use platform. Our powerful AI predicts your likely response to platinum-based chemotherapy, helping you and your doctors make informed decisions. Alongside this, our personalized meal planner crafts nutritious, delicious meal plans supporting your strength and well-being throughout treatment.

But cancer is more than just medicine — it’s about community. That’s why Oncolink offers a secure space for patients to connect, share stories, and support each other through chat and video calls. 
</p>
<p>You’re never alone on your journey with Oncolink by your side.
          </p>
        </div>
      </div>

      {/* Move the buttons OUTSIDE the overlay */}
      <div className="button-container-horizontal">
        <button className="nav-button" onClick={() => window.location.href = "/diagnostic"}>AI Diagnostic</button>
        <button
  className="nav-button"
  onClick={() => window.location.href = "https://lungcancer-cn-4.onrender.com"}
>
  Communications
</button>
        <button className="nav-button" onClick={() => window.location.href = "/meal-planner/planner.html"}>Meal Planner</button>

      </div>
    </div>
  );
}

export default App;