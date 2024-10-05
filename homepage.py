import pandas as pd
import streamlit as st
from io import StringIO
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Homepage:
    def __init__(self, filepath):
        try:
            # Attempt to read the dataset
            self.data = pd.read_csv(filepath)
        except FileNotFoundError:
            st.error(f"File '{filepath}' not found. Please check the path.")
            self.data = pd.DataFrame()  # Empty DataFrame as fallback

    def dataset(self):
        st.header("Dataset Report")
        st.subheader("1. Dataset")
        if not self.data.empty:
            st.write(self.data)
        else:
            st.warning("No data to display. Please check the file path or file format.")

    def summaryInfo(self):
        st.subheader("2.Summary Information")
        if not self.data.empty:
            buffer = StringIO()
            self.data.info(buf=buffer)
            st.text(buffer.getvalue())
            st.write(f"Null values\t\t= {self.data.isnull().sum().sum()}")
            st.write(f"Duplicated values= {self.data.duplicated().sum()}")

        else:
            st.warning("No data to summarize. Please load the dataset.")


class Preprocessing:
    def __init__(self, filepath):
        try:
            self.data = pd.read_csv(filepath)
        except FileNotFoundError:
            st.error(f"File '{filepath}' not found. Please check the path.")
            self.data = pd.DataFrame()  # Empty DataFrame as fallback

    def dataset(self):
        copy = self.data.copy()
        st.header("Dataset Overview")
        st.write(copy.head(5))
        st.header("1.Species normalization")
        st.write("""
        The unique values for the species column are: \n
             1. iris-setosa\n
             2. iris-versicolor\n
             3. iris-virginica\n
        Remove the iris segment and remain with the specific species type      
        """)
        copy["species"] = copy["Species"].str.split(pat="-").str[1]
        code = """df["species"] = df["Species"].str.split(pat = "-").str[1]"""
        st.code(code, language="python")
        st.write(copy.head(3))
        st.write("Drop the initial species column and Id column")
        copy.drop(columns=["Species", "Id"], axis=1, inplace=True)
        code = """df.drop(columns = ["Species", "Id"], axis = 1, inplace = True)"""
        st.code(code, language="python")
        st.write(copy.head(3))

    def labelling(self):
        self.data["species"] = self.data["Species"].str.split(pat="-").str[1]
        self.data.drop(columns=["Species", "Id"], axis=1, inplace=True)
        st.subheader("Labelling and Scaling")
        st.write("MinMaxScaler is used to convert the data input in ranges 0 - 1")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.preprocessing import MinMaxScaler, LabelEncoder
            scaler = MinMaxScaler()
            numerical_cols = self.data.select_dtypes(include="number").columns
            self.data[numerical_cols] = scaler.fit_transform(self.data[numerical_cols])
            le = LabelEncoder()
            for column in self.data.select_dtypes(include=["object"]):
                self.data[column] = le.fit_transform(self.data[[column]])
        code = """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.preprocessing import MinMaxScaler, LabelEncoder
            scaler = MinMaxScaler()
            for column in df.select_dtypes(include = ["number"]):
                 df[column] = scaler.fit_transform(df[[column]])    
            le = LabelEncoder()
            for column in df.select_dtypes(include = ["object"]):
                 df[column] = le.fit_transform(df[[column]])"""
        st.code(code, language="python")
        st.write("Data after scaling and labelling")
        st.write(self.data)
        return self.data, scaler, le


class Modelling:
    def __init__(self, data, scaler):
        self.data = data
        self.scaler = scaler
        st.write(self.data)

    def modelling(self):
        from sklearn.model_selection import train_test_split

        X = self.data.drop(columns=["species"])
        y = self.data["species"]

        from imblearn.over_sampling import SMOTE

        imbalanced = SMOTE(sampling_strategy="minority")

        X_sm, y_sm = imbalanced.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, stratify=y_sm)

        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(criterion="entropy")

        model.fit(X_train, y_train)
        st.header("Model Metrics ")

        st.subheader("1.Confusion Matrix")
        st.write("""
        The following is a heatmap of the confusion matrix in relation to predictions and target values\n
        The diagonal is the correct predictions while the offsets are the incorrect predictions
        """)
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))
        sns.heatmap(cm, annot=True, ax=axs)
        plt.xlabel("predictions")
        plt.ylabel("truth")
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader("2.Classification Report")
        st.write("""Using the f1 score precision recall we can evaluate the model performance\n""")

        st.subheader("Precision Formula:")
        st.latex(r'''
        \text{Recall} = \frac{TP}{TP + FP}
        ''')

        st.subheader("Recall Formula:")
        st.latex(r'''
        \text{Recall} = \frac{TP}{TP + FN}
        ''')
        st.subheader("F1 score Formula")
        # F1 Score formula
        st.latex(r'''
        F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
        ''')

        st.write("Where:")
        st.write("- TP = True Positives")
        st.write("- FP = False Positives")
        st.write("- FN = False Negatives")

        report_df = pd.DataFrame(classification_report(y_test, predictions, output_dict=True))
        st.table(report_df)

        st.subheader("3.Score")
        st.write("""
        The score is a comparison of the true target values for testing set and the prediction of the testing set
        """)
        st.text(f"Accuracy score of the model = {np.round(model.score(X_test, y_test) * 100, 2)}%")

        st.header("Prediction")
        # Get inputs from the user
        try:
            SepalLength = float(st.slider("Sepal Length: ", min_value=0, max_value=10, value=4))
            SepalWidth = float(st.slider("Sepal Width: ", min_value=0, max_value=10, value=4))
            PetalLength = float(st.slider("Petal Length: ", min_value=0, max_value=10, value=4))
            PetalWidth = float(st.slider("Petal Width: ", min_value=0, max_value=10, value=4))
        except ValueError:
            st.error("Please input valid numbers for  all fields.")
            return None

        # Ensure inputs are not empty or invalid
        if not (SepalLength and SepalWidth and PetalLength and PetalWidth):
            st.warning("All inputs must be filled with valid numbers.")
            return None
        # Style customization
        centered_button = """
            <style>
            .centered_button{
                display: flex,
                justify-content: center,
                align-items: center,
                padding: 130px 130px,
            }
            </style>
        """
        # Display button with css
        st.markdown(centered_button, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 0.6, 1])
        species_map = {
            0: "iris - setosa",
            1: "iris - versicolor",
            2: "iris - virginica"
        }
        with col2:
            if st.button("predict"):
                # Prepare the data for scaling (reshape to 2D array)
                data = np.array([[SepalLength, SepalWidth, PetalLength, PetalWidth]])

                # Scale the input features using the pre-fitted scaler
                scaled_data = self.scaler.transform(data)  # This should now work without errors

                predictions = model.predict(scaled_data)

                # Map predictions to species
                species_map = {
                    0: "iris - setosa",
                    1: "iris - versicolor",
                    2: "iris - virginica"
                }

        st.success(f"Predicted species: {species_map[predictions[0]]}")



