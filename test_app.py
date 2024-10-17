import streamlit as st
import numpy as np
import pandas as pd

# Title
st.title("Simple Streamlit App")

# Text input
name = st.text_input("Enter your name", "Streamlit User")
st.write(f"Hello, {name}!")

# Slider input
slider_val = st.slider("Choose a number", 0, 100, 50)
st.write(f"You chose: {slider_val}")

# Display a simple chart
st.write("Here's a random line chart:")
data = pd.DataFrame(
    np.random.randn(10, 3),  # Generate random data
    columns=['A', 'B', 'C']
)
st.line_chart(data)

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df