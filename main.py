
import streamlit as st
from homepage import Homepage
from homepage import Preprocessing, Modelling
st.set_page_config(
    page_title="Flower Classification",
    page_icon="flower.png"
)
df = st.file_uploader(label="Upload csv file")


if df:
    st.sidebar.image("lotus.png", width=200)
    options = st.sidebar.selectbox(
        "Sections",
        [] + ["Home Page", "Data Preprocessing", "Statistical modelling"])

    st.sidebar.success("Select an option above.")

    if "scaling" not in st.session_state:
        st.session_state.scaling = None
    if "data" not in st.session_state:
        st.session_state.data = None

    # When the user selects "Home Page"
    if options == "Home Page":
        # Initialize the Homepage class with the dataset path
        home = Homepage(df)  # Update this to the correct path if needed
        home.dataset()
        home.summaryInfo()
    if options == "Data Preprocessing":
        prep = Preprocessing(df)
        prep.dataset()
        data, scaler, labeller = prep.labelling()
        st.session_state.scaling = scaler
        st.session_state.data = data
    if options == "Statistical modelling":
        if st.session_state.data is not None and st.session_state.scaling is not None:
            model = Modelling(st.session_state.data, st.session_state.scaling)
            model.modelling()
        else:
            st.warning("Complete data preprocessing first")
else:
    st.write("Upload csv file")
