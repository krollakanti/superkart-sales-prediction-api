
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
    # Handle Product Category Code
    # --------------------------------------------------

    # Raw single-product input
    if 'Product_Id' in data.columns:
        data['Product_Category_Code'] = data['Product_Id'].str[:2]

    # Already feature-engineered batch input
    elif 'Product_Id_char' in data.columns:
        data['Product_Category_Code'] = data['Product_Id_char']

    # --------------------------------------------------
    # Handle Store Age
    # --------------------------------------------------

    # Raw single-product input
    if 'Store_Age_Years' not in data.columns:
        if 'Store_Establishment_Year' in data.columns:
            REFERENCE_YEAR = 2026
            data['Store_Age_Years'] = (
                REFERENCE_YEAR - data['Store_Establishment_Year']
            )

    # --------------------------------------------------
    # Handle Product Type
    # --------------------------------------------------

    # Feature-engineered batch input
    if 'Product_Type' not in data.columns:
        if 'Product_Type_Category' in data.columns:
            data['Product_Type'] = data['Product_Type_Category']

    # --------------------------------------------------
    # Remove columns not used by the model
    # --------------------------------------------------

    data = data.drop(
        columns=[
            'Product_Id',
            'Product_Id_char',
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

    # Get JSON data from request
    product_data = request.get_json()

    # Convert JSON object to DataFrame
    input_data = pd.DataFrame([product_data])

    # Prepare data using the same feature engineering
    # used during model development
    input_data = prepare_input(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return prediction
    return jsonify({
        'Predicted_Product_Store_Sales_Total': round(float(prediction), 2)
    })


# ---------------------------------------------------------
# Batch product sales prediction
# ---------------------------------------------------------
@app.post('/v1/productbatch')
def predict_product_sales_batch():

    # Get uploaded CSV file
    file = request.files['file']

    # Read CSV
    input_data = pd.read_csv(file)

    # Keep original Product_Id for the output
    product_ids = input_data['Product_Id'].tolist()

    # Prepare data
    model_input = prepare_input(input_data)

    # Generate predictions
    predictions = model.predict(model_input)

    # Create output DataFrame
    output_data = pd.DataFrame({
        'Product_Id': product_ids,
        'Predicted_Product_Store_Sales_Total': predictions
    })

    # Round predictions
    output_data['Predicted_Product_Store_Sales_Total'] = (
        output_data['Predicted_Product_Store_Sales_Total'].round(2)
    )

    # Return results as JSON
    return jsonify(
        output_data.to_dict(orient='records')
    )


# ---------------------------------------------------------
# Run Flask application
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
