import streamlit as st
import requests
import json

# App title
st.title("🤖 ML Model Prediction")
st.write("California Housing Price Predictor")

# API endpoint
API_URL = "http://18.118.187.190:8000"

# Sidebar - API health check
with st.sidebar:
    st.header("API Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is healthy")
            st.json(response.json())
        else:
            st.error("❌ API returned error")
    except:
        st.error("❌ Cannot reach API")

# Main content
st.header("Enter Housing Features")

# Create input fields (adjust based on your actual model features)
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Median Income", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
    HouseAge = st.number_input("House Age", min_value=0.0, max_value=60.0, value=20.0, step=1.0)
    AveRooms = st.number_input("Average Rooms", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

with col2:
    Population = st.number_input("Population", min_value=0.0, max_value=40000.0, value=1000.0, step=100.0)
    AveOccup = st.number_input("Average Occupancy", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.0, step=0.1)
    Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.0, step=0.1)

# Predict button
if st.button("🔮 Predict House Price", type="primary"):
    # Prepare input
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }

    # Show loading spinner
    with st.spinner("Calling API..."):
        try:
            # Call API
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                
                # Display result
                st.success(f"### Predicted Price: ${prediction * 100000:,.2f}")
                
                # Show request/response details
                with st.expander("📋 API Details"):
                    st.write("**Request:**")
                    st.json(payload)
                    st.write("**Response:**")
                    st.json(result)
            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. API might be slow or down.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to API. Check if it's running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with FastAPI + Docker + GitHub Actions + AWS EC2")