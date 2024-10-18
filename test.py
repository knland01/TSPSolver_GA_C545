import streamlit as st
import numpy as np
import pandas as pd

# Set the title of the app
st.title("Simple Streamlit App for Testing")

# Create a simple input form
st.header("User Input")
user_input = st.number_input("Enter a number", min_value=0, max_value=100, value=50)

# Button to submit the input
if st.button("Calculate Square and Random Data"):
    # Calculate the square of the input
    square_value = user_input ** 2
    st.write(f"The square of {user_input} is {square_value}")
    
    # Generate a small random dataframe
    random_data = np.random.randn(10, 3)
    df = pd.DataFrame(random_data, columns=['Column A', 'Column B', 'Column C'])
    
    st.write("Here is some random data:")
    st.dataframe(df)

# Show a message at the bottom
st.write("This is a simple Streamlit app for testing purposes.")