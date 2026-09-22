
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask("SuperKart Product Sales Predictor")

# Load the trained model
model = joblib.load("tuned_xgb_model.joblib")


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------
@app.get('/')
def home():
    return "Welcome to the SuperKart Product Sales Prediction API"


# ---------------------------------------------------------
# Helper function for feature engineering
# ---------------------------------------------------------
def prepare_input(data):
    data = data.copy()

    # --------------------------------------------------
    # Product_Id_char: keep as-is for batch input,
    # derive from Product_Id for raw single-product input
    # --------------------------------------------------
    if 'Product_Id_char' not in data.columns and 'Product_Id' in data.columns:
        data['Product_Id_char'] = data['Product_Id'].str[:2]

    # --------------------------------------------------
    # Store_Age_Years: derive from establishment year if missing
    # --------------------------------------------------
    if 'Store_Age_Years' not in data.columns and 'Store_Establishment_Year' in data.columns:
        REFERENCE_YEAR = 2026
        data['Store_Age_Years'] = REFERENCE_YEAR - data['Store_Establishment_Year']

    # --------------------------------------------------
    # Product_Type fallback for batch input
    # --------------------------------------------------
    if 'Product_Type' not in data.columns and 'Product_Type_Category' in data.columns:
        data['Product_Type'] = data['Product_Type_Category']

    # --------------------------------------------------
    # Drop only columns the model was NOT trained on
    # (NOTE: Product_Id_char is retained — the model needs it)
    # --------------------------------------------------
    data = data.drop(
        columns=[
            'Product_Id',
            'Store_Id',
            'Store_Establishment_Year',
            'Product_Type_Category'
        ],
        errors='ignore'
    )

    return data



# ---------------------------------------------------------
# Single product sales prediction
# ---------------------------------------------------------
@app.post('/v1/product')
def predict_product_sales():
    try:
        product_data = request.get_json()
        input_data = pd.DataFrame([product_data])
        input_data = prepare_input(input_data)
        prediction = model.predict(input_data)[0]
        return jsonify({
            'Predicted_Product_Store_Sales_Total': round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500


# ---------------------------------------------------------
# Batch product sales prediction
# ---------------------------------------------------------
@app.post('/v1/productbatch')
def predict_product_sales_batch():
    try:
        # Get uploaded CSV file
        file = request.files['file']

        # Read CSV
        input_data = pd.read_csv(file)

        # Pick whichever identifier column is present
        if 'Product_Id' in input_data.columns:
            product_ids = input_data['Product_Id'].tolist()
        elif 'Product_Id_char' in input_data.columns:
            # No unique per-row id in batch CSV — use row index prefixed by category code
            product_ids = [
                f"{code}_{i}" for i, code in enumerate(input_data['Product_Id_char'])
            ]
        else:
            product_ids = list(range(len(input_data)))

        # Prepare data (same feature engineering as single prediction)
        model_input = prepare_input(input_data)

        # Generate predictions
        predictions = model.predict(model_input)

        # Build output
        output_data = pd.DataFrame({
            'Product_Id': product_ids,
            'Predicted_Product_Store_Sales_Total': predictions.round(2)
        })

        return jsonify(output_data.to_dict(orient='records'))

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500


# ---------------------------------------------------------
# Run Flask application
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
