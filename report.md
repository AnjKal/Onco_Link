# Oncolink Codebase Report

## Overview
Oncolink is a comprehensive, integrated web platform designed to support cancer patients and clinicians by combining advanced AI-driven diagnostics, personalized meal planning, and communication tools. The system aims to empower users with accessible, science-backed resources and a supportive environment throughout their cancer journey. This report provides an in-depth technical and use case analysis of the Oncolink codebase, covering its architecture, features, strengths, and areas for improvement.

## 1. Technical Architecture

### 1.1 Frontend
The frontend of Oncolink is built using React, a popular JavaScript library for building user interfaces. The project is bootstrapped with Create React App, which provides a robust foundation for development, testing, and deployment.

- **Routing:**
  - Utilizes React Router for seamless navigation between different pages, such as login, signup, home, diagnostic, and meal planner.
  - The main routes are defined in `App.js`, mapping URLs to their respective components.
- **Component Structure:**
  - The codebase is organized into reusable components, including `LoginPage`, `SignUpPage`, `ResetPasswordPage`, `HomePage`, and `AIDiagnosticForm`.
  - Each component is responsible for a specific part of the user experience, promoting maintainability and scalability.
- **Styling:**
  - Uses a combination of CSS modules (e.g., `App.css`, `styles.css`) and inline styles for both global and component-level styling.
  - The UI is designed to be modern, clean, and responsive, with attention to accessibility and user experience.
- **Static Assets:**
  - The meal planner feature is implemented as static HTML, CSS, and JavaScript files in the `public/meal-planner` directory.
  - Custom images and backgrounds are used to enhance the visual appeal.

### 1.2 Backend
The backend is implemented using Flask, a lightweight Python web framework. It is divided into multiple services, each responsible for a specific domain of the application.

- **User Management (`app.py`):**
  - Handles user authentication, including signup, login, and password reset.
  - Supports role-based access for doctors and patients, with separate SQLite databases for each user type.
  - Provides RESTful API endpoints for frontend integration.
- **AI Diagnostic Service (`diagnostic_api.py`):**
  - Exposes an endpoint for AI-based diagnostic predictions.
  - Utilizes a fine-tuned transformer model (PyTorch) and a random forest model (scikit-learn) for ensemble predictions.
  - Loads pre-trained models from disk and performs inference on incoming data.
  - Returns probabilistic predictions and labels (e.g., Responder/Non-Responder) to the frontend.
- **Meal Planner Service (`meal_app.py`):**
  - Generates personalized 7-day meal plans based on nutritional data and user requirements.
  - Uses pandas for data manipulation and randomization logic to ensure variety and nutritional balance.
  - Provides extra meal suggestions for flexibility.
  - Exposes a REST API consumed by the static meal planner frontend.

### 1.3 Database
- **SQLite:**
  - Separate databases for doctors and patients (`doctors.db`, `patients.db`) to enhance security and data organization.
  - Stores user credentials and profile information.
  - Simple schema design, suitable for small to medium-scale deployments.

### 1.4 Machine Learning Integration
- **Model Loading:**
  - PyTorch and scikit-learn models are loaded at runtime for fast inference.
  - Models are trained offline and saved as `.pth` and `.pkl` files.
- **Prediction Pipeline:**
  - Input features are validated and preprocessed before being passed to the models.
  - Ensemble logic combines predictions from both models for improved accuracy.
  - Results are returned as JSON for easy consumption by the frontend.

### 1.5 Static Assets and Public Resources
- **Meal Planner:**
  - The meal planner is a standalone static web app that interacts with the backend via REST API.
  - Provides an interactive UI for generating and viewing meal plans.
- **Images and Styles:**
  - Custom backgrounds, images, and CSS enhance the user experience and brand identity.

## 2. Main Features and Use Cases

### 2.1 User Authentication and Roles
Oncolink supports robust user authentication with role-based access control.

- **Signup:**
  - Users can register as either doctors or patients.
  - Doctors provide name, age, email, password, and designation.
  - Patients provide name, age, email, password, and sex.
  - Duplicate email checks prevent multiple accounts with the same email.
- **Login:**
  - Users log in with their email, password, and role.
  - Credentials are validated against the appropriate database.
  - Successful login redirects users to their respective dashboards.
- **Password Reset:**
  - Users can reset their password by providing their email, new password, and role.
  - The backend updates the password in the database after validation.
- **Security:**
  - Role-based separation ensures that doctors and patients have access only to relevant features.
  - Passwords are stored in the database (note: consider hashing for production).

### 2.2 AI Diagnostic Tool
The AI Diagnostic Tool is a core feature designed to assist clinicians in predicting patient responses to treatment.

- **Data Input:**
  - Doctors input patient data, including gene expression levels and treatment response.
  - The form is dynamically generated based on required input features.
- **Prediction:**
  - The backend processes the input and runs it through an ensemble of ML models.
  - Returns probabilistic predictions and a final label (Responder/Non-Responder).
- **Clinical Support:**
  - Provides actionable insights to clinicians, supporting evidence-based decision-making.
  - Results are displayed in a user-friendly format in the UI.
- **Extensibility:**
  - The architecture allows for easy integration of additional models or features in the future.

### 2.3 Meal Planner
The Meal Planner feature empowers patients to maintain a healthy diet tailored to their needs.

- **Personalized Plans:**
  - Generates a 7-day meal plan based on nutritional data and calorie limits.
  - Ensures variety by avoiding repetition of meals across days.
- **Backend Logic:**
  - Uses a knapsack-like algorithm to select meals that meet calorie and score thresholds.
  - Arranges meals to avoid repetition and maximize nutritional value.
- **Frontend Experience:**
  - Patients interact with a static web page to generate and view their meal plans.
  - Extra meal suggestions are provided for added flexibility.
  - Responsive design ensures usability across devices.
- **Data Source:**
  - Nutritional data is stored in a CSV file (`nutri2.csv`) and processed by the backend.

### 2.4 Communications
- **External Platform:**
  - Provides a link to an external communications platform for patient-doctor or peer support.
  - Facilitates real-time communication and community building.
- **Future Expansion:**
  - The architecture supports potential integration of in-app messaging or chat features.

## 3. Technical Strengths

- **Separation of Concerns:**
  - Clear division between frontend (React), backend (Flask APIs), and static assets.
- **Extensibility:**
  - Modular design allows for easy addition of new features, such as more AI tools or user roles.
- **Security:**
  - Role-based authentication and separate databases for user types enhance security.
- **User Experience:**
  - Modern, responsive UI with clear navigation and helpful feedback.
- **AI Integration:**
  - Real-world application of machine learning for clinical support.
- **Scalability:**
  - The use of RESTful APIs and modular components supports scaling and future enhancements.

## 4. Potential Use Cases

Oncolink is designed to address a variety of real-world needs in cancer care:

- Empowering cancer patients with personalized, actionable information.
- Supporting clinicians with AI-driven diagnostic tools for better treatment planning.
- Providing holistic care through nutrition planning and healthy meal suggestions.
- Facilitating communication between patients, doctors, and support networks.
- Enabling research and data collection for continuous improvement of care.

## 5. Detailed File and Module Breakdown

### 5.1 Frontend Files
- **`src/App.js`:** Main entry point for routing and component integration.
- **`src/components/`:** Contains authentication and role selection components.
- **`src/HomePage.js`:** Dashboard with navigation to main features.
- **`src/AIDiagnosticForm.js`:** Form for AI diagnostic input and result display.
- **`public/meal-planner/`:** Static files for the meal planner feature.
- **`src/styles.css`, `App.css`:** Styling for the application.

### 5.2 Backend Files
- **`app.py`:** User management API (signup, login, reset password).
- **`diagnostic_api.py`:** AI diagnostic API using PyTorch and scikit-learn.
- **`meal_app.py`:** Meal plan generation API using pandas and randomization.
- **`nutri2.csv`:** Nutritional data for meal planning.
- **Model files:** Pre-trained ML models for diagnostics.

### 5.3 Database Files
- **`doctors.db`, `patients.db`:** SQLite databases for user data.

### 5.4 Static and Public Files
- **`public/index.html`:** Main HTML entry point for the React app.
- **`public/meal-planner/planner.html`, `meal-plan.html`:** Static meal planner pages.

## 6. Recommendations and Areas for Improvement

- **API Unification:**
  - Consider unifying the API endpoints under a single backend for easier deployment and maintenance.
- **Security Enhancements:**
  - Implement password hashing and authentication tokens (e.g., JWT) for improved security.
  - Use HTTPS for all communications.
- **Error Handling:**
  - Add more robust error handling and user feedback in the frontend and backend.
- **Testing:**
  - Increase test coverage for both frontend and backend components.
- **Documentation:**
  - Expand documentation for API endpoints, data models, and deployment procedures.
- **Scalability:**
  - Consider migrating to a more scalable database (e.g., PostgreSQL) for larger deployments.
- **User Experience:**
  - Expand the communications feature for in-app messaging and notifications.
  - Enhance accessibility for users with disabilities.
- **DevOps:**
  - Automate deployment and CI/CD pipelines for smoother updates.

## 7. Conclusion

Oncolink represents a robust, extensible platform for cancer care, integrating advanced AI, personalized nutrition, and communication tools. Its modular architecture, clear separation of concerns, and focus on user experience make it a strong foundation for future growth. By addressing the recommendations above, Oncolink can further enhance its impact and scalability, supporting patients and clinicians in their journey toward better outcomes.

---
This report summarizes the technical and functional aspects of the Oncolink codebase as of June 2025. For further details, refer to the codebase and documentation.
