import numpy as np
from logger import logger


def train_model(df_stock):
    """
    Trains a model on the preprocessed stock data.

    Prefers XGBoost if available; otherwise falls back to a pure-numpy
    linear-regression solution to avoid requiring compiled C runtimes.

    Returns:
        tuple: (date_test, y_test, y_pred, mse, mae, r2, rmse, features, model)
    """
    try:
        target = 'Close'
        features = [col for col in df_stock.columns if col not in ['Date', 'Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 'Close', 'Adj Close']]
        X = df_stock[features]
        y = df_stock[target]

        if len(X) < 10:
            logger.error("Not enough data to train model.")
            raise ValueError("Insufficient data for training.")

        # Split chronologically
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        date_test = df_stock['Date'][split_index:]

        logger.info(f"Training data: {len(X_train)} samples, Test data: {len(X_test)} samples")

        # Try to use XGBoost if available (fast and accurate). If import fails
        # (common on Windows without VC++ runtimes), fall back to a numpy-based
        # linear regression to keep the app runnable.
        try:
            import xgboost as xgb

            logger.info("Using XGBoost for training.")
            xgb_regressor = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=13,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )

            xgb_regressor.fit(X_train, y_train)
            y_pred = xgb_regressor.predict(X_test)
            model = xgb_regressor

        except Exception as e:
            logger.warning(f"XGBoost unavailable or failed to import: {e}. Falling back to numpy linear regression.")

            # Numpy linear regression (closed-form, regularized could be added)
            X_train_arr = X_train.values.astype(float)
            X_test_arr = X_test.values.astype(float)
            y_train_arr = y_train.values.astype(float)

            # Add bias column
            X_train_aug = np.hstack([np.ones((X_train_arr.shape[0], 1)), X_train_arr])
            X_test_aug = np.hstack([np.ones((X_test_arr.shape[0], 1)), X_test_arr])

            # Solve for weights using pseudo-inverse
            weights = np.linalg.pinv(X_train_aug) @ y_train_arr
            y_pred = X_test_aug @ weights
            model = None

        # Compute metrics using numpy (avoid requiring sklearn)
        mse = float(np.mean((y_test.values - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_test.values - y_pred)))
        ss_res = float(np.sum((y_test.values - y_pred) ** 2))
        ss_tot = float(np.sum((y_test.values - np.mean(y_test.values)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        rmse = float(np.sqrt(mse))

        logger.info(f"Model metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}, RMSE: {rmse:.2f}")

        return date_test, y_test, y_pred, mse, mae, r2, rmse, features, model

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise
