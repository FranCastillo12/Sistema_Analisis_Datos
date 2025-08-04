import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def clustering(df: pd.DataFrame):
    print(df)