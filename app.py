import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'train.csv'
df = pd.read_csv(data_path)

# Streamlit app title
st.title("Employee Data Visualization")

# Sidebar for user selection
st.sidebar.header("Visualization Options")
selected_chart = st.sidebar.selectbox("Select a chart to display:", 
                                      ["Gender Distribution", "Region Distribution", "Education Distribution", "Promoted Employees by Region"])

# Gender Distribution
if selected_chart == "Gender Distribution":
  st.header("NUMBER OF MALES AND FEMALES IN THE DATASET")
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'])
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.set_title('Number of Males and Females')
        st.pyplot(fig)

# Region Distribution
elif selected_chart == "Region Distribution":
    if 'region' in df.columns:
        region_counts = df['region'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(region_counts.index, region_counts.values, color='green')
        ax.set_xlabel('Region')
        ax.set_ylabel('Number of People')
        ax.set_title('Number of People by Region')
        plt.xticks(rotation=90)
        st.pyplot(fig)

# Education Distribution
elif selected_chart == "Education Distribution":
    if 'education' in df.columns:
        education_counts = df['education'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(education_counts.values, labels=education_counts.index, autopct='%1.1f%%', colors=['gold', 'lightblue', 'lightgreen'])
        ax.set_title('Education Distribution')
        st.pyplot(fig)

# Promoted Employees by Region
elif selected_chart == "Promoted Employees by Region":
    if 'is_promoted' in df.columns and 'region' in df.columns:
        promoted_regions = df[df['is_promoted'] == 1]['region'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(promoted_regions.index, promoted_regions.values, color='purple')
        ax.set_xlabel('Region')
        ax.set_ylabel('Number of Promoted Employees')
        ax.set_title('Regions Where Employees Are Promoted')
        plt.xticks(rotation=90)
        st.pyplot(fig)
      
# Recruitment Channel Distribution
elif selected_chart == "Recruitment Channel Distribution":
    if 'recruitment_channel' in df.columns:
        recruitment_counts = df['recruitment_channel'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(recruitment_counts.values, labels=recruitment_counts.index, autopct='%1.1f%%', colors=['red', 'blue', 'green'])
        ax.set_title('Recruitment Channel Distribution')
        st.pyplot(fig)
