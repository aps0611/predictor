import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =====================
# Load Trained Models
# =====================
with open("xgb_lang.pkl", "rb") as f:
    xgb_model_lang = pickle.load(f)

with open("xgb_math.pkl", "rb") as f:
    xgb_model_math = pickle.load(f)

# =====================
# Subject Labels
# =====================
lang_labels = {
    0: "Lang_Beginner",
    1: "Lang_Letter",
    2: "Lang_Word",
    3: "Lang_Paragraph",
    4: "Lang_Story",
}

math_labels = {
    0: "Maths_Beginner",
    1: "Maths_NR1",
    2: "Maths_NR2",
    3: "Maths_Sub",
    4: "Maths_Div",
}

# =====================
# Cluster Labels
# =====================
cluster_labels = {
    0: "Math Challenged",
    1: "Balance Mix",
    2: "Foundationally Challenged",
    3: "Language Challenged",
    4: "Progressed",
    9: "Data Not Available",
}

# =====================
# Helpers
# =====================
def prepare_features(df_batch, subject, subject_labels):
    """Prepare batch-level distribution features for model input."""
    class_size = len(df_batch)
    print(df_batch)
    dist = (
        df_batch.groupby(f"bl_{subject}")
        .size()
        .reindex(subject_labels.keys(), fill_value=0)
        .reset_index()
        .rename(columns={0: "count"})
    )
    print(dist)
    dist_dict = {
        subject_labels[level]: count / class_size
        for level, count in zip(dist[f"bl_{subject}"], dist["count"])
    }
    dist_dict["class_size"] = class_size
    print(dist_dict)


    for k, v in dist_dict.items():
        df_batch[k] = v

    return df_batch


def predict_students(df_batch, model, subject, subject_labels):
    """Predict student endline performance using trained model."""
    df_prepared = prepare_features(df_batch.copy(), subject, subject_labels)
    X = df_prepared.drop(columns=["Batch Code_f"], errors="ignore")
    preds = model.predict(X)
    df_prepared[f"pred_el_{subject}"] = preds
    return df_prepared


def compute_cluster(math_dist, lang_dist, class_size):
    """Compute cluster profile based on ratios."""
    print(math_dist)
    MBL2_Non_Arithmetic_Ratio = (
        (math_dist["Maths_Beginner"] + math_dist["Maths_NR1"] + math_dist["Maths_NR2"])
    )
    RBL2_Non_Reader_Ratio = (
        (lang_dist["Lang_Beginner"] + lang_dist["Lang_Letter"] + lang_dist["Lang_Word"])
    )

    if MBL2_Non_Arithmetic_Ratio < 0.2 and RBL2_Non_Reader_Ratio < 0.2:
        return 4
    elif (
        0.65 <= MBL2_Non_Arithmetic_Ratio <= 1
        and 0.8 <= RBL2_Non_Reader_Ratio <= 1
    ):
        return 2
    elif 0.7 <= MBL2_Non_Arithmetic_Ratio <= 1 and RBL2_Non_Reader_Ratio < 0.8:
        return 0
    elif 0 <= MBL2_Non_Arithmetic_Ratio <= 0.65 and 0.7 <= RBL2_Non_Reader_Ratio <= 1:
        return 3
    elif (
        0.2 <= MBL2_Non_Arithmetic_Ratio <= 0.7
        and 0.2 <= RBL2_Non_Reader_Ratio <= 0.7
    ):
        return 1
    elif (
        0.2 <= MBL2_Non_Arithmetic_Ratio <= 0.7
        and 0 <= RBL2_Non_Reader_Ratio <= 0.2
    ):
        return 1
    elif (
        0 <= MBL2_Non_Arithmetic_Ratio <= 0.2
        and 0.2 <= RBL2_Non_Reader_Ratio <= 0.7
    ):
        return 1
    else:
        return None


def plot_distribution(baseline_dist, endline_dist, subject_labels, title):
    """Plot 100% stacked bar baseline vs endline distributions."""
    df_plot = pd.DataFrame(
        {
            "Baseline": [
                baseline_dist[label] for label in subject_labels.values()
            ],
            "Endline": [
                endline_dist[label] for label in subject_labels.values()
            ],
        },
        index=subject_labels.values(),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    df_plot.T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="tab20",
    )
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)

# =====================
# Streamlit UI
# =====================
st.title("📊 Student Learning Prediction & Batch Profiles")

batch_code = st.text_input("Enter Batch Code", "B001")
num_students = st.number_input("Number of Students", min_value=1, step=1)

students_data = []

st.subheader("Enter Student Data")
for i in range(num_students):
    st.markdown(f"### Student {i+1}")
    attendance = st.number_input(
        f"Attendance (Student {i+1})",
        min_value=0,
        max_value=100,
        step=1,
        key=f"att_{i}",
    )
    bl_lang = st.selectbox(
        f"Baseline Language (Student {i+1})",
        options=list(lang_labels.keys()),
        format_func=lambda x: lang_labels[x],
        key=f"lang_{i}",
    )
    bl_math = st.selectbox(
        f"Baseline Mathematics (Student {i+1})",
        options=list(math_labels.keys()),
        format_func=lambda x: math_labels[x],
        key=f"math_{i}",
    )
    students_data.append(
        {
            "Batch Code_f": batch_code,
            "attendance": attendance,
            "bl_language": bl_lang,
            "bl_mathematics": bl_math,
        }
    )

if st.button("Predict Batch Profile"):
    df_new = pd.DataFrame(students_data)

    # Language predictions
    df_lang_pred = predict_students(
        df_new[["Batch Code_f", "attendance", "bl_language"]],
        xgb_model_lang,
        subject="language",
        subject_labels=lang_labels,
    )

    # Maths predictions
    df_math_pred = predict_students(
        df_new[["Batch Code_f", "attendance", "bl_mathematics"]],
        xgb_model_math,
        subject="mathematics",
        subject_labels=math_labels,
    )

    # Merge predictions
    df_final = df_new.copy()
    df_final["pred_el_language"] = df_lang_pred["pred_el_language"]
    df_final["pred_el_mathematics"] = df_math_pred["pred_el_mathematics"]

    # Compute baseline distributions
    baseline_lang_dist = df_new["bl_language"].map(lang_labels).value_counts(normalize=True).reindex(lang_labels.values(), fill_value=0).to_dict()
    baseline_math_dist = df_new["bl_mathematics"].map(math_labels).value_counts(normalize=True).reindex(math_labels.values(), fill_value=0).to_dict()
    print(baseline_lang_dist)
    # Compute endline distributions
    endline_lang_dist = df_final["pred_el_language"].map(lang_labels).value_counts(normalize=True).reindex(lang_labels.values(), fill_value=0).to_dict()
    endline_math_dist = df_final["pred_el_mathematics"].map(math_labels).value_counts(normalize=True).reindex(math_labels.values(), fill_value=0).to_dict()

    # Clusters
    baseline_cluster = compute_cluster(baseline_math_dist, baseline_lang_dist, num_students)
    endline_cluster = compute_cluster(endline_math_dist, endline_lang_dist, num_students)

    st.subheader("Cluster Profiles")
    st.success(f"Baseline Cluster: {cluster_labels.get(baseline_cluster, 'Unknown')}")
    st.success(f"Predicted Endline Cluster: {cluster_labels.get(endline_cluster, 'Unknown')}")

    # Visuals
    st.subheader("📊 Distributions")
    plot_distribution(baseline_lang_dist, endline_lang_dist, lang_labels, "Language Distribution (Baseline vs Endline)")
    plot_distribution(baseline_math_dist, endline_math_dist, math_labels, "Mathematics Distribution (Baseline vs Endline)")

    # Student-level results
    st.subheader("Student Predictions")
    st.dataframe(df_final)
