# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the dataset
df = pd.read_csv('D:\AI Projects\AI Diabetes Prediction\diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup 👨🏻‍⚕️')
st.sidebar.header('Patient Data')

# Function to get user input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


st.subheader('User Data:')
# Get patient data
user_data = user_report()

# Convert DataFrame to Markdown
markdown_text = user_data.to_markdown()

#Toshow the user_data in table 
st.table(user_data)



# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)




# OUTPUT
st.subheader('Your Result:📋 ')
if user_result[0] == 0:
    st.markdown('<div style="border: 2px solid green; padding: 10px; color: green; font-weight: bold; width:180px;">You are not Diabetic</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="border: 2px solid red; padding: 10px; color: red; font-weight: bold; width:180px;">You are Diabetic</div>', unsafe_allow_html=True)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')



# Graph - 1(line chart )

st.subheader('User Data Chart:📈')
# Draw a line chart according to user_data
plt.figure(figsize=(10, 6))

# Plot each parameter
plt.plot(user_data.columns, user_data.values.flatten(), marker='o')

# Labeling each line
for i, col in enumerate(user_data.columns):
    plt.text(i, user_data[col].values[0], col, ha='center', va='bottom')

# Add labels and title
plt.xlabel('Parameters')
plt.ylabel('Values')
plt.title('User Data')

# Show the plot
st.pyplot(plt)





# # 
# # Graph - 2(scatterplot) 

# # Draw a line chart according to user_data with different colors for different parameters
# plt.figure(figsize=(10, 6))

# # # Define colors for each parameter
# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

# # Plot each parameter with a different color
# for i, col in enumerate(user_data.columns):
#     plt.plot(col, user_data[col].values[0], marker='o', color=colors[i], label=col)

# # Add legend
# plt.legend()

# # # Add labels and title
# plt.xlabel('Parameters')
# plt.ylabel('Values')
# plt.title('User Data')

# # Show the plot
# st.pyplot(plt)











