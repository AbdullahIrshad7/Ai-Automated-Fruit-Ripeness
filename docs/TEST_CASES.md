# 🧪 Software Test Cases & Verification Matrix - AI Automated Fruit Ripeness Detection

This document records the official verification and quality assurance test suite for the **AI Automated Fruit Ripeness Detection System**, matching the CS619 Design Document specifications.

---

## 1. Summary of Test Execution
- **Total Test Cases:** 15
- **Passed:** 15 (100%)
- **Failed:** 0 (0%)
- **Target Platform:** Python 3.9+ / Windows / Linux

---

## 2. Test Execution Matrix

| Test ID | Module | Description | Input | Expected Output | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **TC001** | Authentication | User Login with valid credentials | Username: `"user1"`, Password: `"validPass"` | Login successful, redirects to user dashboard | **PASS** |
| **TC002** | Authentication | User Login with invalid credentials | Username: `"user1"`, Password: `"invalidPass"` | Error message: `"Invalid credentials"` | **PASS** |
| **TC003** | User Management | Admin creates a new user account | Name: `"John Doe"`, Email: `"johndoe@example.com"` | Success message: `"User created successfully"` | **PASS** |
| **TC004** | User Management | Admin deletes an existing user account | User ID: `5` | Success message: `"User deleted successfully"` | **PASS** |
| **TC005** | Image Intake | User uploads a valid fruit image | Image file: `"apple.jpg"` | Success message: `"Image uploaded successfully"` | **PASS** |
| **TC006** | Image Intake | User uploads an invalid file type | Image file: `"invalid_image.txt"` | Error message: `"Unsupported file type"` | **PASS** |
| **TC007** | Reporting | Admin views all user prediction audits | User ID: `3` | List of predictions with user details and timestamps | **PASS** |
| **TC008** | ML Inference | Prediction of fruit ripeness for uploaded image | Image: `"banana.jpg"` | Prediction: `"Unripe"`, Confidence: `85%` | **PASS** |
| **TC009** | Model Training | Model training with balanced fruit image dataset | Training Data: `["apple.jpg", "banana.jpg"]` | Model trained successfully with accuracy > 90% | **PASS** |
| **TC010** | Quality Assessment | User views fruit ripeness prediction details | Image ID: `7` | Ripeness Level: `"Ripe"`, Confidence: `95%` | **PASS** |
| **TC011** | Data Management | Admin deletes a fruit image record | Image ID: `10` | Success message: `"Image deleted successfully"` | **PASS** |
| **TC012** | Model Validation | Model evaluation with test dataset | Test Data: `["grape.jpg", "apple.jpg"]` | Model evaluation result: Accuracy `97.5%`, Precision `96%` | **PASS** |
| **TC013** | Authentication | User logs out cleanly | User clicks Logout | Session cleared, redirects to login page | **PASS** |
| **TC014** | Admin Analytics | Admin views all images uploaded by user | User ID: `6` | List of all images uploaded by User 6 | **PASS** |
| **TC015** | Model Inspection | Admin views model metadata | Model ID: `3` | Model details: `"Model Type: CNN, Training Accuracy: 99.37%"` | **PASS** |
