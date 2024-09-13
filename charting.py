import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mapper = {
    'gpt-4-turbo-preview': 'GPT-4 Turbo',
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini-2024-07-18': 'GPT-4o Mini',
    'claude-3-opus-20240229': 'Claude 3 Opus',
    'claude-3-5-sonnet-20240620': 'Claude 3.5 Sonnet',
    'gemini-1_5-pro': 'Gemini 1.5 Pro',
    'gemini-1_0-pro': 'Gemini 1.0 Pro',
    'gemini-1_5-pro-exp-0801': 'Gemini 1.5 Pro Ex',
    'mistral-large-latest': 'Mistral Large 2',
    'open-mixtral-8x22b': 'Mistral 8x22B',
    'meta_llama3-70b-instruct-v1_0': 'Llama 3 70B',
    'meta_llama3-1-70b-instruct-v1_0': 'Llama 3.1 70B',
    'command-r': 'Command R',
    'command-r-plus': 'Command R Pro',
    'Meta-Llama-3-1-405B-Instruct-jjo_eastus_models_ai_azure_com': 'Llama 3.1 405B',
    'Meta-Llama-3-1-70B-Instruct-ostu_eastus_models_ai_azure_com': 'Llama 3.1 70B',
}

def define_data(final_stats: pd.DataFrame):
    ## Define the data
    # models = ["Human level*", "GPT-4 Turbo", "Claude 3 Opus", "Mistral Large", "Gemini Pro 1.5",
    #           "Gemini Pro 1.0", "Llama 3 70B", "Mistral 8x22B"]
    # mean_scores = [80, 38, 33, 30, 29, 27, 21, 16]
    # lower_bounds = [10, 16, 15, 15, 15, 15, 13, 11]
    # upper_bounds = [10, 16, 15, 15, 15, 15, 13, 11]

    final_stats['model'] = final_stats['model'].map(mapper).fillna(final_stats['model'])
    final_stats.loc[-1] = {
        'model': 'Human level*',
        'mean_score': 86,
        'std_dev_score': 0,
        'z_interval_error': 0,
        'ci_lower': 93,
        'ci_upper': 78,
    }
    final_stats = final_stats.sort_values(by='mean_score', ascending=False)

    models = final_stats['model'].to_list()
    mean_scores = final_stats['mean_score'].to_list()
    lower_bounds = final_stats['ci_lower'].to_list()
    upper_bounds = final_stats['ci_upper'].to_list()

    data = {
        "Model": models,
        "Average": mean_scores,
        "Confidence Interval Low": lower_bounds,
        "Confidence Interval High": upper_bounds,
    }
    return pd.DataFrame(data)


def create_performance_chart(final_stats: pd.DataFrame, title="LLM Linguistic Benchmark Performance",
                             highlight_models=None):
    if highlight_models is None:
        highlight_models = []

    df = define_data(final_stats)
    # Create a basic barplot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Different colors for different models
    colors = ['skyblue' if model not in highlight_models else 'orange' for model in df["Model"]]
    barplot = sns.barplot(data=df, x="Model", y="Average", palette=colors, errorbar=None)

    # Shade the first bar with black cross lines
    for i, bar in enumerate(barplot.patches):  # Loop through the bars
        if df["Model"][i] == "Human level*":  # Check if it is the Human level bar
            bar.set_hatch('///')  # Apply hatching

    # Add confidence intervals as vertical lines with caps
    capwidth = 0.1  # Width of the cap lines
    for i, model in enumerate(df["Model"]):
        plt.plot([i, i], [df["Confidence Interval Low"][i], df["Confidence Interval High"][i]],
                 color='grey', lw=1)
        # Add horizontal caps
        plt.plot([i - capwidth / 2, i + capwidth / 2],
                 [df["Confidence Interval Low"][i], df["Confidence Interval Low"][i]],
                 color='grey', lw=1)
        plt.plot([i - capwidth / 2, i + capwidth / 2],
                 [df["Confidence Interval High"][i], df["Confidence Interval High"][i]],
                 color='grey', lw=1)

    plt.title(title, fontsize=18)
    plt.xlabel("", fontsize=14)
    plt.ylabel("Average Score (%)", fontsize=14)
    plt.xticks(rotation=60, fontsize=14)
    plt.tight_layout()

    return barplot, plt