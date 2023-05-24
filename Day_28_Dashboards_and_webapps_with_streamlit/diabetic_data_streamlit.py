# PIMA indians diabetic data analysis and ploting with Streamlit

# import libs
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Pil import Image


# loading dataset
df = pd.read_csv("D:\Python ka Chilla Dobara\Day_28_Dashboards_and_webapps_with_streamlit\diabetes.csv")
st.write(df)