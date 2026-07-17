import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request
from logger import logger
from utils import load_data, preprocess_data, validate_stock_symbol
from models import train_model

app = Flask(__name__)

# --- Load Data Globally Once ---
data_path = 'data.csv'
df, stock_symbols = load_data(data_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_stock = request.form.get('stock', 'MMM')  # Default to MMM

    logger.info(f"Request received for stock: {selected_stock}")

    if df.empty:
        logger.error("Data not loaded.")
        return render_template('index.html',
                               stock_symbols=stock_symbols,
                               selected_stock=selected_stock,
                               error="Error: Could not load data.csv")

    if not validate_stock_symbol(selected_stock, stock_symbols):
        return render_template('index.html',
                               stock_symbols=stock_symbols,
                               selected_stock=selected_stock,
                               error=f"Invalid stock symbol '{selected_stock}'.")

    df_stock = preprocess_data(df, selected_stock)

    if df_stock.empty or len(df_stock) < 10:  # Check for enough data after preprocessing
        logger.warning(f"Not enough data for stock '{selected_stock}'.")
        return render_template('index.html',
                               stock_symbols=stock_symbols,
                               selected_stock=selected_stock,
                               error=f"Not enough data for stock '{selected_stock}' to train model.")

    try:
        date_test, y_test, y_pred, mse, mae, r2, rmse, features, model = train_model(df_stock)
    except Exception as e:
        logger.error(f"Model training failed for '{selected_stock}': {str(e)}")
        return render_template('index.html',
                               stock_symbols=stock_symbols,
                               selected_stock=selected_stock,
                               error="Error training model. Please try again.")

    # --- Create Matplotlib Graph ---
    plt.figure(figsize=(18, 9))
    plt.plot(date_test, y_test.values, label='Actual Price', color='navy', marker='.', alpha=0.7)
    plt.plot(date_test, y_pred, label='Predicted Price', color='darkorange', linestyle='--', marker='.', alpha=0.8)
    plt.title(f'Stock Price Prediction vs. Actual for {selected_stock}', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price (USD)', fontsize=14)
    plt.legend(fontsize=12)

    # Save plot to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    logger.info(f"Successfully generated prediction for '{selected_stock}'.")
    return render_template('index.html',
                           stock_symbols=stock_symbols,
                           selected_stock=selected_stock,
                           plot_data=plot_data,
                           mse=f"{mse:.2f}",
                           mae=f"{mae:.2f}",
                           r2=f"{r2:.2f}",
                           rmse=f"{rmse:.2f}",
                           error=None)

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True)
