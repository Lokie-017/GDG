import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive

# Load the dataset
drive.mount('/content/drive')
data_path = '/content/drive/MyDrive/train.csv'
df = pd.read_csv(data_path)


# Plot number of males and females
if 'gender' in df.columns:
    gender_counts = df['gender'].value_counts()
    plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'])
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Number of Males and Females')
    plt.show()

# Plot number of people by region
if 'region' in df.columns:
    region_counts = df['region'].value_counts()
    plt.bar(region_counts.index, region_counts.values, color='green')
    plt.xlabel('Region')
    plt.ylabel('Number of People')
    plt.title('Number of People by Region')
    plt.xticks(rotation=90)
    plt.show()

# Plot education distribution in pie chart
if 'education' in df.columns:
    education_counts = df['education'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(education_counts.values, labels=education_counts.index, autopct='%1.1f%%', colors=['gold', 'lightblue', 'lightgreen'])
    plt.title('Education Distribution')
    plt.show()

# Plot regions where employees are promoted in bar chart
if 'is_promoted' in df.columns and 'region' in df.columns:
    promoted_regions = df[df['is_promoted'] == 1]['region'].value_counts()
    plt.bar(promoted_regions.index, promoted_regions.values, color='purple')
    plt.xlabel('Region')
    plt.ylabel('Number of Promoted Employees')
    plt.title('Regions Where Employees Are Promoted')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
