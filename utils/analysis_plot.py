import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# https://matplotlib.org/stable/api/markers_api.html
MARKER_BY_BASELINE = {
    "Lexical": "s", #"X",
    "Named Entity": "s", # "X",
    "Document": "s"
}


# Manually ordering names because this is how they will eventually show up in the legend
ordered_group_names = [
    'NN DeID',  
    'IDF',
    'IDF (table-aware)', 
    'Document', 
    'Lexical',
    'Named Entity'
]

experiment_to_group_name = {
    "lexical": "Lexical", 
    "nn_deid_biencoder_table": "NN DeID",
    "idf": "IDF",
    "idf_table": "IDF (table-aware)",
    "named_entity": "Named Entity",
    "document": "Document",
}

PARETO_COLORS = dict(
    zip(ordered_group_names, sns.color_palette("hls", len(ordered_group_names)))
)

def make_pareto_plot(df: pd.DataFrame, pdf_filename: str):
    """Makes the Pareto plot from Figure 2."""
    sns.set_theme(context="paper", style="white")
    x_column = 'masking_percentage'
    y_column = 'was_reidentified'

    xlabel = "% Words masked"
    ylabel = "Reidentification % (Ensemble)"

    assert {'experiment_name', 'masking_percentage'} <= set(df.columns), f'invalid columns {df.columns}'

    sns.set(style="white", font_scale = 1.4)
    plt.figure(figsize=(9,6))
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    
    df["group_name"] = df["experiment_name"].map(experiment_to_group_name.get)
    df["marker"] = df.apply(lambda row: MARKER_BY_BASELINE.get(row["group_name"], "o"), axis=1)
    g1 = sns.lineplot(
        data=df,
        x=x_column,
        y=y_column,
        hue='group_name',
        palette=PARETO_COLORS,
        legend=True,
    )
    # plt.yscale('log', base=2)
    # yticks = [0.01, 0.1, 1.0]
    yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(ticks=yticks, labels=yticks)
    
    # hacky line of code to make sure the squares are plotted at the end, and therefore show up after everything else
    for marker_type, marker_size in [('o', 100), ('s', 180)]:
        df_with_this_marker = df[
            df["marker"] == marker_type
        ]
        g2 = sns.scatterplot(
            data=df_with_this_marker,
            x=x_column,
            y=y_column,
            hue="group_name",
            palette=PARETO_COLORS,
            s=marker_size,
            marker=marker_type,
            linewidth=0.0,
            legend=False
        )
        
        if marker_type == 's':
            plt.setp(g2.lines, zorder=100)
            plt.setp(g2.collections, zorder=100, label="")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [ordered_group_names.index(label) for label in labels]
    handles, labels = [handles[idx] for idx in order], [labels[idx] for idx in order]
    plt.legend(handles, labels, bbox_to_anchor=(0.84, 1.05), framealpha=1.0)
    plt.tight_layout()
    plt.savefig(pdf_filename, dpi=300)
