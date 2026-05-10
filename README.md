This project builds a two-stage hybrid machine learning pipeline for NFL in-game win probability estimation. Most existing models apply a single classifier across all game situations — we argue that a blowout in the first quarter and a one-score game with two minutes left are fundamentally different prediction problems and should be modeled separately.

We first use K-Means clustering to segment plays into five interpretable game states (Comfortable Lead, Close/Late Game, Big Deficit, Early Neutral, Red Zone Drive), then train a dedicated Gradient Boosted Tree classifier within each cluster. The pipeline is evaluated against three baselines — Logistic Regression, a global GBT without clustering, and the Pythagorean Win Expectancy heuristic — across Accuracy, F1, ROC-AUC, and Brier Score.

Built with scikit-learn, pandas, matplotlib, and seaborn. Compatible with real NFL data via the nfl-data-py package.
