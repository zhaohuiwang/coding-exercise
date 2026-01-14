
import numpy as np
from pathlib import Path
import torch


def find_project_root(
    start: Path | None = None,
    *, # everything after this must be passed as a keyword argument
    markers: tuple[str, ...] = (),
    dirname: str | None = None,
) -> Path:
    """
    Find the project root directory by searching upward in the filesystem.

    Starting from `start` (or the current working directory if not provided),
    this function walks up through parent directories until it finds a
    directory that matches one of the following conditions:

    - Contains at least one file or directory listed in `markers`
      (e.g. "pyproject.toml", ".git")
    - Has a directory name equal to `dirname`

    At least one of `markers` or `dirname` must be provided.

    Parameters
    ----------
    start : Path | None, optional
        The directory to start searching from. If None, the current working
        directory is used. The path is resolved to an absolute path before
        searching.

    markers : tuple[str, ...], keyword-only, optional
        A tuple of file or directory names that indicate the project root.
        If any marker exists in a directory, that directory is considered
        the project root.

    dirname : str | None, keyword-only, optional
        The expected name of the project root directory. If a directory with
        this name is encountered while searching upward, it is returned as
        the project root.

    Returns
    -------
    Path - The resolved absolute path of the detected project root directory.

    Raises
    ------
    ValueError - If neither `markers` nor `dirname` is provided.

    FileNotFoundError - If no matching project root is found before reaching the filesystem root.

    Examples
    --------
    Find project root using marker files:
    >>> find_project_root(markers=("pyproject.toml", ".git"))

    Find project root using directory name:
    >>> find_project_root(dirname="my_project")

    Hybrid search (recommended):
    >>> find_project_root(
            start=Path(__file__).parent,
            dirname="my_project",
            markers=("pyproject.toml", ".git")
        )
    """
    if not markers and not dirname:
        raise ValueError("Provide at least one of `markers` or `dirname`")

    if start is None:
        start = Path.cwd()

    start = start.resolve()

    for path in [start, *start.parents]:
        if dirname and path.name == dirname:
            return path

        if markers and any((path / m).exists() for m in markers):
            return path

    raise FileNotFoundError(
        f"Project root not found starting from {start} "
        f"(dirname={dirname}, markers={markers})"
    )


class EarlyStopping:
    def __init__(
            self,
            patience: int=7,
            verbose: bool=False,
            delta: float=0,
            save_model: bool=False,
            path: str | Path ='best_model.pth'
            ):
        """
        EarlyStopping if place to the end of each epoch cycle, monitors val_loss (validation loss) in each epoch (or set of epochs through DataLoader). When the criteria defined by patience and delta is met, early_stop is set to True to signal the break in the epoch iteration. save_checkpoint() is optional and maybe removed.   
        Args:
            patience (int): Number of epochs with no improvement before stopping.
            verbose (bool): If True, prints messages on improvement.
            delta (float): Minimum change to qualify as an improvement.
            save_model (bool): Whether to save the best model.
            path (str | Path): Path to save the model checkpoint.

        Example:
        early_stopping = EarlyStopping(patience=3)
        for epoch in range(cfg.optuna.n_epochs_per_trial):
            model.train()
            ...
            model.eval()
            ...
            val_loss = ...
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_model = save_model
        self.path = Path(path)

    def __call__(self, val_loss, model):
        score = - val_loss # lower val_loss is better

        if self.best_score is None:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(model, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(model, val_loss)
            self.counter = 0

    def save_checkpoint(self, model, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f"Validation loss decreased"
                  f"({self.val_loss_min:.6f} --> {val_loss:.6f})."
                  "Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
