
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import shap


def missing_summary(
    df: pd.DataFrame, sort: bool = True, per_float: int = 2
) -> pd.DataFrame:
    """
    Summarize missing values in a DataFrame and optionally display in a notebook
    or print to terminal/logs.
    Args:
        df: Input DataFrame
        sort: Sort by missing percentage descending
        per_float: Decimal pint to round the percent of missing
    Returns:
        pd.DataFrame: Missing value summary
    """
    summary = (
        df.isna()
        .sum()
        .to_frame(name="missing_count")
        .assign(
            missing_percent=lambda x: (x["missing_count"] / len(df)) * 100,
            dtype=df.dtypes.astype(str),
        )
        .reset_index(names="column")
    )

    # Sort
    if sort:
        summary = summary.sort_values(
            "missing_percent",
            ascending=False,
            kind="mergesort",
        )
    summary = summary.reset_index(drop=True)

    summary["missing_percent"] = summary["missing_percent"].round(per_float)

    return summary

def normalize_columns_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names in place:
    - lowercase
    - strip leading/trailing whitespace
    - replace internal whitespace with underscores
    """
    # df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)
    # return df

def plot_cross_correlation_heatmap(
    df: pd.DataFrame,
    target_vars: list[str],
    group_col: str | None = None,
    groups: list | None = None,
    corr_threshold: float | None = None,
    title_prefix: str = "Heatmap of Correlations",
    figsize: tuple[int, int] = (8, 3),
    cmap: str = "coolwarm",
    save_path: str | Path | None = None,
    save_csv: bool = False,  # <- New argument to save cross_corr to CSV
) -> None:
    """
    Plots cross-correlation heatmaps between target variables and numeric variables.
    Supports optional group splitting, correlation thresholding, saving figures,
    and saving cross-correlation data to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_vars : list[str]
        List of target variable column names.
    group_col : str | None, optional
        Column to split data into groups. Default None.
    groups : list | None, optional
        Specific groups to plot. Default None (all unique groups).
    corr_threshold : float | None, optional
        Minimum absolute correlation to display. Default None (no threshold).
    title_prefix : str, optional
        Prefix for plot titles.
    figsize : tuple[int, int], optional
        Figure size.
    cmap : str, optional
        Colormap for heatmap.
    save_path : str | Path | None, optional
        Directory to save figures and/or CSV. Default None.
    save_csv : bool, optional
        If True, saves the cross-correlation data to CSV. Default False.
    """

    def _plot_heatmap(data: pd.DataFrame, suffix: str = "") -> None:
        numeric_df = data.select_dtypes(include=["number"])
        numeric_vars = [c for c in numeric_df.columns if c not in target_vars]

        all_cols = target_vars + numeric_vars
        corr_matrix = numeric_df[all_cols].corr()
        cross_corr = corr_matrix.loc[target_vars, numeric_vars]

        if corr_threshold is not None:
            cross_corr = cross_corr.where(cross_corr.abs() >= corr_threshold)

        # Save cross-correlation CSV if requested
        if save_csv and save_path is not None:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename_csv = f"{title_prefix.replace(' ', '_').lower()}{suffix}.csv"
            cross_corr.to_csv(save_dir / filename_csv)

        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            cross_corr,
            annot=True,
            cmap=cmap,
            fmt=".2f",
            linewidths=0.5,
            mask=cross_corr.isna(),
        )
        plt.title(f"{title_prefix}{suffix}")
        plt.xlabel("Numeric variables")
        plt.ylabel("Target variables")
        plt.tight_layout()

        # Save figure if requested
        if save_path is not None:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename_fig = f"{title_prefix.replace(' ', '_').lower()}{suffix}.png"
            plt.savefig(save_dir / filename_fig, dpi=300)

        plt.show()

    # ---- Handle group_col splitting ----
    if group_col:
        if groups is None:
            groups = df[group_col].dropna().unique()
        for g in groups:
            subset = df[df[group_col] == g]
            _plot_heatmap(subset, suffix=f"_{group_col}_{g}")
    else:
        _plot_heatmap(df)


def plot_shap_summary_multi_target(
    model: object,
    X: pd.DataFrame,
    target_names: list[str],
    feature_names: list[str] | None = None,
    explainer_type: str = "tree",
    sample_size: int = 100,
    random_state: int = 42,
    max_display: int = 20,
    top_k: int = 10,
    figsize: tuple[int, int] = (5, 5),
    save_path: str | Path | None = None,
    export_shap_values: bool = False,
    plot_type: str = "dot",
) -> None:
    """
    Computes SHAP values and plots summary plots for multiple targets, robustly handling
    different SHAP output formats. Supports CSV export, top-k features, multiple SHAP plot types,
    combined subplots for matplotlib-based plots, and unified saving logic.

    Parameters
    ----------
    model : object
        Trained model (tree-based, kernel, or linear compatible with SHAP).
    X : pd.DataFrame
        Input features for SHAP analysis.
    target_names : list[str]
        Names of the targets in a multi-output model.
    feature_names : list[str] | None, optional
        Feature names. Defaults to X.columns.
    explainer_type : str, optional
        Type of SHAP explainer: "tree", "kernel", or "linear". Default "tree".
    sample_size : int, optional
        Number of rows to sample from X for SHAP computation. Default 100.
    random_state : int, optional
        Random state for sampling. Default 42.
    max_display : int, optional
        Maximum number of features to display in each summary or bar plot. Default 20.
    top_k : int, optional
        Number of top features to highlight in CSV export. Default 10.
    figsize : tuple[int, int], optional
        Figure size for matplotlib-based plots. Default (5, 5).
    save_path : str | Path | None, optional
        Directory to save plots or CSVs. Default None.
    export_shap_values : bool, optional
        If True, exports SHAP values to CSV files. Default False.
    plot_type : str, optional
        Type of SHAP plot to generate: "dot" (default), "bar", "waterfall".

    Returns
    -------
    None
    """

    # Sample input data
    X_explain: pd.DataFrame = X.sample(sample_size, random_state=random_state)
    if feature_names is None:
        feature_names = X.columns.tolist()

    # Ensure save_path is a Path object
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Initialize explainer
    if explainer_type.lower() == "tree":
        explainer = shap.TreeExplainer(model)
    elif explainer_type.lower() == "kernel":
        explainer = shap.KernelExplainer(model.predict, X_explain)
    elif explainer_type.lower() == "linear":
        explainer = shap.LinearExplainer(model, X_explain)
    else:
        raise ValueError(
            "Unsupported explainer_type. Choose 'tree', 'kernel', or 'linear'."
        )

    # Compute SHAP values
    shap_values_all: list[np.ndarray] | np.ndarray = explainer.shap_values(X_explain)

    # Determine function to extract target-specific SHAP values
    if isinstance(shap_values_all, list):
        n_targets = len(shap_values_all)

        def get_shap_for_target(i: int) -> np.ndarray:
            return shap_values_all[i]

    elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
        n_targets = shap_values_all.shape[2]

        def get_shap_for_target(i: int) -> np.ndarray:
            return shap_values_all[:, :, i]

    else:
        n_targets = 1

        def get_shap_for_target(i: int) -> np.ndarray:
            return shap_values_all

    # Loop over each target
    for i, target in enumerate(target_names):
        shap_values_target: np.ndarray = get_shap_for_target(i)

        explanation: shap.Explanation = shap.Explanation(
            values=shap_values_target,
            base_values=(
                explainer.expected_value[i]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value
            ),
            data=X_explain.values,
            feature_names=feature_names,
        )

        # Export SHAP values to CSV
        if export_shap_values and save_path is not None:
            shap_df: pd.DataFrame = pd.DataFrame(
                shap_values_target, columns=feature_names
            )
            shap_df.to_csv(save_path / f"shap_values_{target}.csv", index=False)
            top_features: pd.Series = (
                shap_df.abs().mean().sort_values(ascending=False).head(top_k)
            )
            top_features.to_csv(save_path / f"shap_top{top_k}_{target}.csv")

        # Plotting based on plot_type
        plot_type_lower = plot_type.lower()
        matplotlib_plot = plot_type_lower in ("dot", "bar", "waterfall")

        plt.figure(figsize=figsize)
        if plot_type_lower == "bar":
            shap.plots.bar(explanation, max_display=max_display, show=False)
        elif plot_type_lower in ("dot", "violin"):
            shap.summary_plot(
                explanation,
                max_display=max_display,
                plot_type=plot_type_lower,
                show=False,
            )
        elif plot_type_lower == "waterfall":
            shap.plots.waterfall(explanation[0], show=False)
        else:
            raise ValueError(
                f'Unsupported plot_type: {plot_type}. Choose from "dot", "bar", "waterfall".'
            )
        plt.title(f"SHAP Summary Plot ({plot_type_lower}) for {target}")
        plt.tight_layout()

        # --- Unified saving for all matplotlib plots ---
        if save_path is not None and matplotlib_plot:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            filename = f"shap_{target}_{plot_type_lower}.png"
            plt.savefig(save_path / filename, dpi=300)

        # --- Unified show for matplotlib plots ---
        if matplotlib_plot:
            plt.show()