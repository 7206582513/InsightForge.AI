import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("static/charts", exist_ok=True)

def save_chart(fig, filename):
    output_dir = os.path.join("static", "charts")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

def generate_charts(df):
    chart_paths = []
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    chart_paths.append(save_chart(fig, "correlation_heatmap.png"))

    for col in df.select_dtypes(include='number').columns[:5]:
        fig = plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        chart_paths.append(save_chart(fig, f"hist_{col}.png"))

    for col in df.select_dtypes(include='object').columns[:3]:
        fig = plt.figure()
        sns.countplot(y=df[col])
        plt.title(f"Count of {col}")
        chart_paths.append(save_chart(fig, f"bar_{col}.png"))

    if df.iloc[:, -1].nunique() <= 10:
        fig = plt.figure()
        sns.countplot(x=df.iloc[:, -1])
        plt.title("Target Class Distribution")
        chart_paths.append(save_chart(fig, "target_distribution.png"))

    return chart_paths