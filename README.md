# 📊 Customer Churn Prediction Studio

An interactive, enterprise-grade full-stack machine learning workbench for analyzing, validating, and forecasting customer churn. This solution leverages **SMOTE (Synthetic Minority Over-sampling Technique)** to resolve class imbalance, trains multi-algorithm classifiers on-the-fly in client-side TypeScript, and generates high-touch outreach playbooks powered by **Gemini 3.5 Flash**.

---

## 🎨 Professional Polish UI preview

### 🖥️ Dashboard & Heuristics Overview
![Dashboard Overview](screenshots/dashboard_overview.png)
*Figure 1: Complete view of the Churn predictions, dynamic KPI metrics, and gradient training convergence curve.*

### ⚖️ Preprocessing: SMOTE Class Balancer
![SMOTE Visualizer](screenshots/smote_viz.png)
*Figure 2: Real-time 2D Scatter plot showing synthetic customer nodes generated to resolve cohort imbalancing.*

### 🧪 Dynamic Sandbox and Retention Playground
![Interactive Playground](screenshots/sandbox_simulator.png)
*Figure 3: Tweaking variables in real-time to predict churn risk and drafting enterprise retention playbooks instantly.*

---

## 🚀 Key Features

1. **Synthetic Sample Generation (SMOTE)**:
   - Interpolates synthetic minority customer cancelation records along line segments of their nearest neighbors ($k$-NN).
   - Mitigates default-class prediction bias (over-categorizing everyone as "Loyal") to boost actual **Recall (Sensitivity)**.
2. **Interactive Algorithmic Workbench**:
   - **Logistic Regression**: High-stability predictive probability curve optimized with L2 regularization and gradient descent.
   - **Decision Tree**: Binary segmentation tree calculating sequential Gini entropy impurity reductions.
   - **Neural Network (MLP)**: Input layer ($\times 7$), hidden dense layer, and output activation powered by backpropagation.
3. **Enterprise Analytics & Diagnostics**:
   - Live-updating **Confusion Matrix** showing FP, TN, FN, and TP ratios.
   - Comprehensive **ROC Curve** displaying False Positive vs. True Positive Sweeps with an estimated **AUC Score**.
   - **Feature Importance Magnitude Charts** ranking key driver coefficients.
4. **AI-Powered Playbook Outreach**:
   - Proxies cohort risk factors back to **Gemini 3.5 Flash** server-side.
   - Emits tactical account diagnostics alongside custom personalized email outreach scripts.

---

## 🏗️ Full-Stack Technical Architecture
