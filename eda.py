import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "dataset/cleaned_study_planner_dataset.csv"
)

print("Shape:", df.shape)

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\nColumns:")
print(df.columns)

print("\nSummary Statistics:")
print(df.describe())

# =====================================================
# CORRELATION MATRIX
# =====================================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(14,10))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# =====================================================
# STUDY HOURS DISTRIBUTION
# =====================================================

if "study_hours_per_day" in df.columns:

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["study_hours_per_day"],
        bins=20,
        kde=True
    )

    plt.title("Study Hours Distribution")
    plt.xlabel("Study Hours Per Day")

    plt.show()

# =====================================================
# EXAM SCORE DISTRIBUTION
# =====================================================

if "exam_score" in df.columns:

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["exam_score"],
        bins=20,
        kde=True
    )

    plt.title("Exam Score Distribution")
    plt.xlabel("Exam Score")

    plt.show()

# =====================================================
# GPA DISTRIBUTION
# =====================================================

if "previous_gpa" in df.columns:

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["previous_gpa"],
        bins=20,
        kde=True
    )

    plt.title("GPA Distribution")

    plt.show()

# =====================================================
# ATTENDANCE VS EXAM SCORE
# =====================================================

if (
    "attendance_percentage" in df.columns
    and
    "exam_score" in df.columns
):

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        data=df,
        x="attendance_percentage",
        y="exam_score"
    )

    plt.title(
        "Attendance Percentage vs Exam Score"
    )

    plt.show()

# =====================================================
# STUDY HOURS VS EXAM SCORE
# =====================================================

if (
    "study_hours_per_day" in df.columns
    and
    "exam_score" in df.columns
):

    plt.figure(figsize=(8,5))

    sns.regplot(
        data=df,
        x="study_hours_per_day",
        y="exam_score"
    )

    plt.title(
        "Study Hours vs Exam Score"
    )

    plt.show()

# =====================================================
# STRESS LEVEL VS EXAM SCORE
# =====================================================

if (
    "stress_level" in df.columns
    and
    "exam_score" in df.columns
):

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x="stress_level",
        y="exam_score",
        data=df
    )

    plt.title(
        "Stress Level vs Exam Score"
    )

    plt.show()

# =====================================================
# MOTIVATION LEVEL VS EXAM SCORE
# =====================================================

if (
    "motivation_level" in df.columns
    and
    "exam_score" in df.columns
):

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x="motivation_level",
        y="exam_score",
        data=df
    )

    plt.title(
        "Motivation Level vs Exam Score"
    )

    plt.show()

print("\nEDA Completed Successfully")