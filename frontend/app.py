
import streamlit as st
import pandas as pd
import requests


# ---------------------------------------------------------
# Backend URL
# ---------------------------------------------------------
BACKEND_URL = "http://backend:7860"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide"
)


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------
st.title("🛒 SuperKart Product Sales Prediction App")

st.write(
    "Enter the product and store details below to predict "
    "the total product-store sales."
)


# =========================================================
# SINGLE PRODUCT PREDICTION
# =========================================================

st.header("Single Product Sales Prediction")


# Product information
col1, col2 = st.columns(2)

with col1:

    Product_Id = st.text_input(
        "Product ID",
        value="FD6114"
    )

    Product_Weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=10.0
    )

    Product_Sugar_Content = st.selectbox(
        "Product Sugar Content",
        [
            "Low Sugar",
            "Regular",
            "No Sugar"
        ]
    )

    Product_Allocated_Area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        value=0.10
    )

    Product_Type = st.selectbox(
        "Product Type",
        [
            "Meat",
            "Snack Foods",
            "Hard Drinks",
            "Dairy",
            "Canned",
            "Soft Drinks",
            "Health and Hygiene",
            "Baking Goods",
            "Bread",
            "Breakfast",
            "Frozen Foods",
            "Fruits and Vegetables",
            "Household",
            "Seafood",
            "Starchy Foods",
            "Others"
        ]
    )

    Product_MRP = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=150.0
    )


with col2:

    Store_Id = st.text_input(
        "Store ID",
        value="OUT049"
    )

    Store_Establishment_Year = st.number_input(
        "Store Establishment Year",
        min_value=1900,
        max_value=2026,
        value=1999,
        step=1
    )

    Store_Size = st.selectbox(
        "Store Size",
        [
            "High",
            "Medium",
            "Low"
        ]
    )

    Store_Location_City_Type = st.selectbox(
        "Store Location City Type",
        [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )

    Store_Type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Supermarket Type1",
            "Supermarket Type2",
            "Food Mart"
        ]
    )


# ---------------------------------------------------------
# Create JSON payload
# ---------------------------------------------------------

product_data = {

    "Product_Id": Product_Id,

    "Product_Weight": Product_Weight,

    "Product_Sugar_Content": Product_Sugar_Content,

    "Product_Allocated_Area": Product_Allocated_Area,

    "Product_Type": Product_Type,

    "Product_MRP": Product_MRP,

    "Store_Id": Store_Id,

    "Store_Establishment_Year": Store_Establishment_Year,

    "Store_Size": Store_Size,

    "Store_Location_City_Type": Store_Location_City_Type,

    "Store_Type": Store_Type
}


# ---------------------------------------------------------
# Single prediction button
# ---------------------------------------------------------

if st.button(
    "Predict Product Store Sales",
    type="primary"
):

    try:

        response = requests.post(
            f"{BACKEND_URL}/v1/product",
            json=product_data
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result[
                "Predicted_Product_Store_Sales_Total"
            ]

            st.success(
                "Prediction completed successfully!"
            )

            st.metric(
                "Predicted Product Store Sales",
                f"{prediction:,.2f}"
            )

        else:

            st.error(
                f"Prediction failed. "
                f"Backend returned status code "
                f"{response.status_code}."
            )

            st.json(response.json())

    except requests.exceptions.RequestException as e:

        st.error(
            f"Unable to connect to the prediction API: {e}"
        )


# =========================================================
# BATCH PREDICTION
# =========================================================

st.header("Batch Product Sales Prediction")

st.write(
    "Upload a CSV file containing multiple product-store "
    "records to generate sales predictions."
)


uploaded_file = st.file_uploader(
    "Upload Product CSV",
    type=["csv"]
)


if uploaded_file is not None:

    # Display uploaded data
    input_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(
        input_df,
        use_container_width=True
    )


    # -----------------------------------------------------
    # Batch prediction
    # -----------------------------------------------------

    if st.button(
        "Predict Batch Sales",
        type="primary"
    ):

        # Reset file pointer
        uploaded_file.seek(0)

        try:

            response = requests.post(
                f"{BACKEND_URL}/v1/productbatch",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "text/csv"
                    )
                }
            )


            if response.status_code == 200:

                results = response.json()

                st.success(
                    "Batch predictions completed successfully!"
                )

                # Convert results to DataFrame
                results_df = pd.DataFrame(results)

                # Display predictions
                st.subheader("Prediction Results")

                st.dataframe(
                    results_df,
                    use_container_width=True
                )


                # -------------------------------------------------
                # Download predictions
                # -------------------------------------------------

                csv_data = results_df.to_csv(
                    index=False
                ).encode("utf-8")


                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_data,
                    file_name="superkart_sales_predictions.csv",
                    mime="text/csv"
                )


            else:

                st.error(
                    f"Batch prediction failed. "
                    f"Backend returned status code "
                    f"{response.status_code}."
                )

                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)


        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to connect to the prediction API: {e}"
            )
