import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(csv_path, save_dir="results/plots"):
    df = pd.read_csv(csv_path)
    sns.barplot(x="Model", y="F1", data=df)
    plt.title("F1-score by Model")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/f1_by_model.png")
    plt.close()
    sns.barplot(x="Model", y="Acc", data=df)
    plt.title("Accuracy by Model")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/acc_by_model.png")
    plt.close()
    sns.barplot(x="SeqLen", y="F1", hue="Model", data=df)
    plt.title("F1 vs Sequence Length")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/f1_by_seq.png")
    plt.close()
