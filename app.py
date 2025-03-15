import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load the dataset
data_path = 'train.csv'
df = pd.read_csv(data_path)

# Streamlit app title
st.title("📊 Employee Data Insights & Visualization")

# Sidebar for user selection
st.sidebar.header("🔍 Select a Visualization")
selected_chart = st.sidebar.selectbox(
    "Choose a chart:", 
    [
        "Gender Distribution", 
        "Region-wise Employee Count", 
        "Education Level Distribution", 
        "Promotion Rate by Region", 
        "Recruitment Channel Preference",
        "Previous Year Rating Distribution",
        "Length of Service Scatter Plot",
        "Department-wise Employee Count"
    ]
)

# Gender Distribution
if selected_chart == "Gender Distribution":
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=gender_counts.index, y=gender_counts.values, palette=["#3498db", "#e74c3c"], ax=ax)
        ax.set_xlabel('Gender', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('📌 Employee Gender Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)

# Region-wise Employee Count
elif selected_chart == "Region-wise Employee Count":
    if 'region' in df.columns:
        region_counts = df['region'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis', ax=ax)
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Employee Count', fontsize=12)
        ax.set_title('🌍 Employee Distribution by Region', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90)
        st.pyplot(fig)

# Education Level Distribution
elif selected_chart == "Education Level Distribution":
    if 'education' in df.columns:
        education_counts = df['education'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(education_counts.values, labels=education_counts.index, autopct='%1.1f%%', colors=['#f1c40f', '#3498db', '#2ecc71'], startangle=140)
        ax.set_title('🎓 Employee Education Level Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)

# Promotion Rate by Region
elif selected_chart == "Promotion Rate by Region":
    if 'is_promoted' in df.columns and 'region' in df.columns:
        promoted_regions = df[df['is_promoted'] == 1]['region'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=promoted_regions.index, y=promoted_regions.values, palette='magma', ax=ax)
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Number of Promoted Employees', fontsize=12)
        ax.set_title('🏆 Promotion Count by Region', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90)
        st.pyplot(fig)

# Recruitment Channel Preference
elif selected_chart == "Recruitment Channel Preference":
    if 'recruitment_channel' in df.columns:
        recruitment_counts = df['recruitment_channel'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(recruitment_counts.values, labels=recruitment_counts.index, autopct='%1.1f%%', colors=['#e74c3c', '#3498db', '#2ecc71'], startangle=140)
        ax.set_title('📢 Recruitment Channel Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)

# Previous Year Rating Distribution
elif selected_chart == "Previous Year Rating Distribution":
    if 'previous_year_rating' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['previous_year_rating'].dropna(), bins=range(1, 7), kde=True, color='blue', edgecolor='black', alpha=0.7, ax=ax)
        ax.set_xlabel("Previous Year Rating", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("⭐ Distribution of Previous Year Ratings", fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 6))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

# Length of Service Scatter Plot
elif selected_chart == "Length of Service Scatter Plot":
    if 'employee_id' in df.columns and 'length_of_service' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['employee_id'], df['length_of_service'], alpha=0.5, color='b')
        ax.set_xlabel("Employee ID", fontsize=12)
        ax.set_ylabel("Length of Service (Years)", fontsize=12)
        ax.set_title("📈 Scatter Plot of Length of Service", fontsize=14, fontweight='bold')
        ax.grid(True)
        st.pyplot(fig)

# Department-wise Employee Count
elif selected_chart == "Department-wise Employee Count":
    if 'department' in df.columns:
        department_counts = df['department'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        department_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
        ax.set_xlabel("Department", fontsize=12)
        ax.set_ylabel("Employee Count", fontsize=12)
        ax.set_title("🏢 Number of Employees in Each Department", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
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
