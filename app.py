from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv,numpy as np,pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()


st.title("Disease_Prediction");
st.divider()






data = pd.read_csv("Training.csv")
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms,indices))

def dosomething(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1,1)).transpose()
    return(dt.predict(user_input_label))





#print(dosomething(['headache','muscle_weakness','puffy_face_and_eyes','mild_fever','skin_rash']))
#prediction = []
#for i in range(1):
#     pred = dosomething(['headache','muscle_weakness'])
#     prediction.append(pred)
#print(prediction)

st.subheader("Who's this test for?")
test = st.radio(
    "",
    ["Myself", "***Friend***", "***Family***"],
    index=None,
)

st.write("You selected:", test)

st.subheader("disease prediction based on symptoms")
with st.form("my_form"):
   def submit():
    prediction = []
    for i in range(1):
        pred = dosomething(options)
        prediction.append(pred)

    return prediction[0][0]
   options = st.multiselect(
    "What are the symptoms you are facing?",list(dictionary.keys())
    )


   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("You're facing with", submit())





st.subheader("EMR Analysis")
uploaded_file = st.file_uploader("Choose an EMR image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded EMR", use_column_width=True)

    # Save the uploaded file temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Upload the file to Google AI
    sample_file = genai.upload_file(path="temp_image.png", display_name="electronic medical record")
    st.write(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")

    # Choose a Gemini API model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

    # Prompt the model
    prompt = "Analyze the given EMR and then detect with the highest probability of that 1 disease she might be facing with given her conditions also don't mention that you are an ai"
    
    if st.button("Analyze EMR"):
        with st.spinner("Analyzing..."):
            response = model.generate_content([sample_file, prompt])
            st.write("Analysis Result:")
            st.write(response.text)

    # Clean up the temporary file
    os.remove("temp_image.png")
