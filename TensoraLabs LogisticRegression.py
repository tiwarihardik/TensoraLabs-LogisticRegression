from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('TensoraLabs - Logistic Regression')


if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'target_categories' not in st.session_state:
    st.session_state.target_categories = None

dataset = st.file_uploader("Choose a CSV File", type='.csv')

if dataset is not None:
    df = pd.read_csv(dataset)
    st.write(df.head())

    target = st.selectbox('Column to be predicted:', df.columns)
    use_column = st.selectbox('Column to be used for prediction:', df.columns.drop(target))

    if st.button('Train Model') and use_column:
        X = pd.get_dummies(df[[use_column]])  
        y = df[target]

        if y.ndim > 1:
            y = y.iloc[:, 0]

        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=4)

        model = LogisticRegression(C=0.04, solver='saga')
        model.fit(train_x, train_y)

        predictions = model.predict(test_x)
        acc = accuracy_score(test_y, predictions)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.target_categories = y.unique().tolist()

        fig, ax = plt.subplots()
        ax.scatter(train_x.iloc[:, 0], train_y, color='blue', label='Actual Data')
        ax.plot(train_x.iloc[:, 0], model.coef_[0][0] * train_x.iloc[:, 0] + model.intercept_[0], color='red', label='Fit Line')
        ax.set_xlabel(use_column)
        ax.set_ylabel(target)
        ax.legend()
        st.pyplot(fig)

        st.write("Model's Accuracy:", acc)

        if acc >= 0.50:
            st.success("✅ High accuracy! The model's predictions are stable and accurate.")
        else:
            st.warning("⚠️ Accuracy could be improved. Try different features or more data.")

        st.info("Model Trained Successfully!")


    if st.session_state.model:
        st.header("Make Predictions")

        if pd.api.types.is_numeric_dtype(df[use_column]):
            value = st.number_input(f"Enter value for {use_column}:")
        else:
            value = st.selectbox(f"Select {use_column}:", df[use_column].unique())

        user_input = {use_column: value}

        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            input_enc = pd.get_dummies(input_df)
            pred = st.session_state.model.predict(input_enc)[0]

            label = pred
            st.success(f"Prediction: {label}")
