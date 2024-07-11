from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv,numpy as np,pandas as pd
import os
import streamlit as st


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



with st.form("my_form"):
   def submit():
    prediction = []
    for i in range(1):
        pred = dosomething(options)
        prediction.append(pred)

    return prediction[0][0]
   st.write("Inside the form")
   options = st.multiselect(
    "What are the symptoms you are facing?",list(dictionary.keys())
    )


   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("You're facing with", submit())










