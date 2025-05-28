from enum import Enum
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib as mpl

# Global matplotlib styling
mpl.rcParams['axes.facecolor'] = '#0a0f2c'      
mpl.rcParams['figure.facecolor'] = '#0a0f2c'    
mpl.rcParams['axes.edgecolor'] = '#0ff'         
mpl.rcParams['axes.labelcolor'] = '#ffffff'    
mpl.rcParams['xtick.color'] = '#ffffff'
mpl.rcParams['ytick.color'] = '#ffffff'
mpl.rcParams['text.color'] = '#ffffff'
mpl.rcParams['axes.titlecolor'] = '#0ff'        
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'sans-serif']

import seaborn as sns
custom_palette = ['#0ff', '#ffffff', '#1ecbe1', '#38f3e0']
sns.set_theme(style="darkgrid", palette=custom_palette)

import matplotlib.pyplot as plt

import io
import urllib, base64
import math

CATEGORICAL_THRESHOLD = 10
DEFAULT_FIGSIZE = (10, 6)

import pandas as pd

class TargetColumnHandler:
    def __init__(self, df: pd.DataFrame, target_column=None):
        self.df = df
        self.target_column = target_column
        self.target_column_name = None

    def identify_target_column(self):
        if not self.target_column:
            self.target_column_name = self.df.columns[-1]
            return

        try:
            column_index = int(self.target_column)
            self.target_column_name = self.df.columns[column_index]
        except ValueError:
            self.target_column_name = self.target_column
        except IndexError:
            raise ValueError(f"Column index {self.target_column} is out of bounds.")

        if self.target_column_name not in self.df.columns:
            raise ValueError(f"Column '{self.target_column_name}' not found in the dataset.")

    def move_target_to_end(self):
        if not self.target_column_name:
            self.identify_target_column()

        cols = [col for col in self.df.columns if col != self.target_column_name] + [self.target_column_name]
        self.df = self.df[cols]

    def get_processed_df(self):
        self.identify_target_column()
        self.move_target_to_end()
        return self.df



class TargetType(Enum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


class BaseAnalyzer:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col

    # Creates and returns a base64 image of a heatmap showing correlations between numeric features in the dataset.
    def plot_correlations(self):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Feature Correlation Heatmap")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_string = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_string

    # Plots and returns a base64 image comparing numeric features to the target using box or scatter plots.
    def plot_feature_vs_target(self, kind="box"):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]

        if not numeric_cols:
            return None

        n_cols = 2
        n_rows = math.ceil(len(numeric_cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=DEFAULT_FIGSIZE)
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            if kind == "box":
                sns.boxplot(x=self.target_col, y=col, data=self.df, ax=ax)
            elif kind == "scatter":
                sns.scatterplot(x=self.df[col], y=self.df[self.target_col], ax=ax)
            ax.set_title(f"{col} vs {self.target_col}")

        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_string = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_string

class ClassificationAnalyzer(BaseAnalyzer):

    def perform_analysis(self):
        plot = {
            "Class Distribution": self.plot_class_distribution(),
            "Feature Relationships": self.pair_plot_dist(),
            "Feature Distributions": self.plot_feature_vs_target(kind="box"),
            "Heat Map": self.plot_correlations()
        }

        return plot

    # Creates and returns a base64 image of a bar chart showing how many times each class appears in the target column.
    def plot_class_distribution(self):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        sns.countplot(x=self.target_col, data=self.df, ax=ax)
        ax.set_title("Class Distribution")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_string = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_string

    # Generates and returns a base64 image of pairwise scatter plots to show relationships between features, colored by target.
    def pair_plot_dist(self):
        sample_df = self.df.sample(n=min(len(self.df), 1000))
        pair_grid = sns.pairplot(sample_df, hue=self.target_col)
        fig = pair_grid.figure

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        img_string = base64.b64encode(buf.read()).decode('utf-8')
        return img_string


class RegressionAnalyzer(BaseAnalyzer):

    def perform_analysis(self):
        plot = {
            "Target Distribution": self.plot_target_distribution(),
            "Feature Distributions": self.plot_feature_vs_target(kind="scatter"),
            "Heat Map": self.plot_correlations()
        }
        return plot

    
    # Creates and returns a base64 image of a smooth histogram showing how the target values are distributed.
    def plot_target_distribution(self):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        sns.histplot(self.df[self.target_col], kde=True, ax=ax)
        ax.set_title("Target Distribution")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_string = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_string


class DataInvestigator:

    def __init__(self, df):
        if 'Unnamed: 0' in df.columns:
            new_df = df.loc[:, df.columns != 'Unnamed: 0']
        else:
            new_df = df.copy()
        self.df = new_df
        self.target_col = df.columns[-1]
        self.target_type = self.check_target_type()
        self.target_data_type = df[self.target_col].dtype

    # Check if the target variable is categorical or continuous based on its data type and number of unique values
    def check_target_type(self):
        unique_values = self.df[self.target_col].nunique()
        dtype = self.df[self.target_col].dtype
        if dtype == "object" or unique_values <= CATEGORICAL_THRESHOLD:
            return TargetType.CATEGORICAL
        return TargetType.CONTINUOUS

    # Determine whether to apply classification or regression based on the data type and nature of the target variable
    def check_model_type(self):
        if self.target_data_type in ['int64', 'float64']:
            if self.target_type == TargetType.CATEGORICAL:  
                return "Classification could be applied as the target variable is numeric with limited unique values."
            else:
                return "Regression should be applied as the target variable is continuous."
        elif self.target_data_type == 'object':
            return "Classification should be applied as the target is categorical."
        else:
            return "Unable to determine the model type based on the target variable. We recommend checking the data visualization of the target variable to better understand its distribution."

    
    # Generates a summary of the dataset including shape, data types, missing values, duplicates, and target column info
    def summarize_dataset(self):
        shape = self.df.shape
        dtypes = self.df.dtypes.to_dict()
        missing = self.df.isnull().sum().to_dict()
        duplicates = self.df.duplicated().sum()
        target_col = self.df.columns[-1]
        unique_vals = self.df[target_col].nunique()
        target_type = self.target_type

        summary = {
            "num_rows": shape[0],
            "num_columns": shape[1],
            "column_types": dtypes,
            "missing_values": missing,
            "duplicate_rows": duplicates,
            "target_column": target_col,
            "target_type": self.target_type,
            "unique_categories": unique_vals if self.target_type == "Categorical" else None
        }

        sentences = [
            f"The dataset contains <b>{summary['num_rows']} rows</b> and <b>{summary['num_columns']} columns</b>.",
            f"The target column is <b>'{summary['target_column']}'</b>, which is likely <b>{target_type.value}</b>.",
            f"There are <b>{duplicates} duplicate rows</b>.",
            f"Missing values are present in <b>{sum(1 for k in missing if missing[k] > 0)} columns</b>.",
        ]

        if target_type == "Categorical":
            sentences.append(f"The target column has <b>{unique_vals} unique categories<b>.")
        
        return summary, sentences

    
    def perform_analysis(self):

        if self.target_type == TargetType.CATEGORICAL:
            analyzer = ClassificationAnalyzer(self.df, self.target_col)
        else:
            analyzer = RegressionAnalyzer(self.df, self.target_col)
        analyzer.perform_analysis()

        return {
            "Model Recommendation" : self.check_model_type(),
            "Dataset Summary" : self.summarize_dataset(),
             "Data Analyser"  : analyzer.perform_analysis()
        }