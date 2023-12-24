from sklearn.datasets import load_iris
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import streamlit as st 

st.markdown(
    '''
    # Simple Iris Flower Prediction
    This app predicts the *Iris* flower type!
    ***
'''
)

st.sidebar.header("User Input Paramaters")

def load_input():
    sepal_length = st.sidebar.slider("sepal length",4.30,7.9,5.4)
    sepal_width = st.sidebar.slider("sepal width",2.,4.4,3.)
    petal_length = st.sidebar.slider("petal length",1.,6.9,3.)
    petal_width = st.sidebar.slider("petal_width",1.,2.5,1.)

    data = {
        "sepal_length": sepal_length,
        "sepal_width" : sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    df = pd.DataFrame(data,index = [0])
    return df

df = load_input()

st.header("User Input")
st.write(df)

rf_clf = RandomForestClassifier(random_state=42)

iris = load_iris()
x = iris.data
y = iris.target
clf = RandomForestClassifier(random_state=42)
clf.fit(x,y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)