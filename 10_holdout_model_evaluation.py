import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_path = 'data/hold_out_evaluation/evaluacion_holdout.csv'
df = pd.read_csv(dataset_path)

# --- 1. Extract category and transformation method if not already present ---
# Category column (you might already have it)
df['category'] = df['dataset'].apply(lambda x: 'important' if 'important' in x else ('pca' if 'pca' in x else 'plain'))

# Transformation method extraction from filename
def extract_transform(dataset_name):
    dataset_name = dataset_name.lower()
    if 'l1' in dataset_name:
        return 'l1'
    elif 'l2' in dataset_name:
        return 'l2'
    elif 'boxcox' in dataset_name:
        return 'boxcox'
    elif 'minmax' in dataset_name:
        return 'minmax'
    elif 'yeojohnson' in dataset_name:
        return 'yeojohnson'
    else:
        return 'training'

df['transformation'] = df['dataset'].apply(extract_transform)

# --- 2. Function to create boxplot grids ---
def plot_boxgrid(df, reference_model, metric='best_r2', save_path='plot.png', grid_title=None):
    # Compute difference to reference model for each category and transformation
    ref_df = df[df['model'] == reference_model][['category', 'transformation', metric]]
    ref_df = ref_df.rename(columns={metric: f'{reference_model}_{metric}'})
    
    # Merge with full dataframe
    comp_df = df.merge(ref_df, on=['category', 'transformation'], how='left')
    comp_df['metric_diff'] = comp_df[metric] - comp_df[f'{reference_model}_{metric}']
    
    # Boxplot grid
    g = sns.catplot(
        data=comp_df,
        x='model',
        y='metric_diff',
        col='category',
        row='transformation',
        kind='box',
        height=3,
        aspect=1.5,
        sharey=False
    )
    g.set_axis_labels("Model", f"{metric} difference vs {reference_model}")
    g.set_titles("{row_name} / {col_name}")
     # Rotate x-axis labels for all facets
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    
    # Set overall title if provided
    if grid_title is not None:
        g.figure.suptitle(grid_title, fontsize=16)
        g.figure.subplots_adjust(top=1)  # Adjust top to make room for the title
    plt.tight_layout()
    plt.savefig(save_path)

# --- 3. Plot ---
hold_out_eval_path = 'data/hold_out_evaluation/'

metrics = ['best_r2', 'best_rmse', 'best_mae']
reference_models = ['linear_regression', 'ridge']

for ref_model in reference_models:
    for metric in metrics:
        # Create a descriptive metric name
        if metric == 'best_r2':
            metric_name = 'R2'
            direction = 'higher is better'
        elif metric == 'best_rmse':
            metric_name = 'RMSE'
            direction = 'lower is better'
        elif metric == 'best_mae':
            metric_name = 'MAE'
            direction = 'lower is better'
        else:
            metric_name = metric
            direction = ''

        # Combine metric name and direction in title
        grid_title = f"{metric_name} Differences vs {ref_model} ({direction})"
        save_path = f"{hold_out_eval_path}boxplot_{ref_model}_{metric}.png"
        
        plot_boxgrid(
            df, 
            reference_model=ref_model, 
            metric=metric, 
            save_path=save_path, 
            grid_title=grid_title
        )
