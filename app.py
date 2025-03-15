import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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


# Streamlit App Title
st.title("Employee Promotion Prediction Using KNN")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Define features and target variable
    X = df.drop(columns=['is_promoted'])  # Adjust column name if needed
    y = df['is_promoted']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a KNN model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Display results
    st.subheader("Model Performance")
    st.write(f"### Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_rep)

    # Identify employees likely to be promoted
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    X_test_df['Predicted_Promotion'] = y_pred
    X_test_df['Actual_Promotion'] = y_test.values

    # Filter employees predicted for promotion
    promoted_employees = X_test_df[X_test_df['Predicted_Promotion'] == 1]

    # Display promoted employees
    st.subheader("Employees Predicted for Promotion")
    st.dataframe(promoted_employees.head(10))

    # Convert to CSV
    promoted_csv = promoted_employees.to_csv(index=False).encode('utf-8')

    # Provide a download button
    st.download_button(
        label="Download Predicted Promotions as CSV",
        data=promoted_csv,
        file_name="predicted_promotions.csv",
        mime="text/csv"
    )
