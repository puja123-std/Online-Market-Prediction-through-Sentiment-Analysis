pip install -r requirements.txt# Stock Market Prediction with Sentiment Analysis

A Flask-based web application that predicts stock prices using machine learning models (XGBoost) incorporating sentiment analysis from news data. The app allows users to select a stock symbol and view predictions along with performance metrics and visualizations.

## Features

- **Stock Selection**: Dropdown to choose from available stock symbols in the dataset.
- **Machine Learning Model**: Uses XGBoost Regressor for price prediction.
- **Sentiment Integration**: Incorporates positive and negative sentiment ratios from news data.
- **Data Preprocessing**: Includes time-based features, lag features, moving averages, and volatility calculations.
- **Visualization**: Displays actual vs. predicted stock prices in an interactive graph.
- **Model Metrics**: Shows MSE, RMSE, MAE, and R² scores for evaluation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/stock-prediction-sentiment.git
   cd stock-prediction-sentiment
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not present, install the following packages:
   - Flask
   - pandas
   - numpy
   - xgboost
   - scikit-learn
   - matplotlib
   - seaborn

4. **Ensure data file**: Place `data.csv` in the root directory. The CSV should contain columns like 'Date', 'Symbol', 'Close', 'News - Positive Sentiment', 'News - Negative Sentiment', 'News - Volume', etc.

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to `http://127.0.0.1:5000/`.

3. **Select a stock symbol** from the dropdown and click "Analyze" to generate predictions and view the graph.

## Data

The application expects a `data.csv` file with historical stock data including:
- Date
- Symbol (stock ticker)
- Close price
- Adjusted Close
- News sentiment data (Positive, Negative, Volume)

Ensure the data is clean and formatted correctly for the preprocessing steps.

## Model Details

- **Algorithm**: XGBoost Regressor
- **Features**: Includes lag close prices, moving averages (7-day, 30-day), volatility, sentiment ratios, and time-based features.
- **Training**: 80% of data for training, 20% for testing (chronological split).
- **Hyperparameters**: Tuned for optimal performance (e.g., max_depth=13, learning_rate=0.05).

## Project Structure

```
.
├── app.py                 # Main Flask application
├── data.csv               # Dataset (not included in repo)
├── static/
│   └── css/
│       └── style.css      # CSS styles
├── templates/
│   └── index.html         # HTML template
├── venv/                  # Virtual environment (optional)
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational purposes only. Stock predictions are not financial advice. Always consult with a financial advisor before making investment decisions.
