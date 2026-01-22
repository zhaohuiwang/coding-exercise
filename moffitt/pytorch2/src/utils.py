
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





def objective(
        # String-based annotation to avoid evaluation. Activate if needed. 
        trial: "optuna.Trial",
        cfg: "ConfigSchema",
        train_loader: "DataLoader", 
        val_loader: "DataLoader", 
        emb_sizes: list[tuple[int, int]],
        device: torch.device,
        task: str = "regression"
        ) -> float:

    # Sample architecture
    #choose how many hidden layers the network has
    n_layers: int = trial.suggest_int('n_layers', *cfg.optuna.layer_range)
    # for each layer, choose how many neurons it has
    hidden_dims: list[int] = [
        trial.suggest_categorical(f'n_units_l{i}', cfg.optuna.units_list) for i in range(n_layers)
    ]
    dropout: float = trial.suggest_float('dropout', *cfg.optuna.dropout_range)
    # syntax: trial.suggest_float(name, low, high, log=False) # Each trial gets a different value

    # Model
    model: nn.Module = DynamicModel(
        emb_sizes, len(cfg.data.num_cols), len(cfg.data.target_cols), hidden_dims, dropout).to(device)
        
    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Loss
    criterion = (
        nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
        )
    # Task type	                Correct loss
    # Multi-class (1 label)     CrossEntropyLoss
    # Multi-label	            BCEWithLogitsLoss
    # Multi-target regression	MSELoss

    # Loop
    # # Early stopping logic - simple code
    # patience = 5
    # best_val = float("inf")
    # epochs_no_improve = 0
    # # Early stopping logic - from a class
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(cfg.optuna.n_epochs_per_trial):
        model.train()
        for xc, xn, y in train_loader:
            # Matches the model and dataloader architecture
            xc, xn, y = xc.to(device), xn.to(device), y.to(device)
            optimizer.zero_grad()
            # model(xc, xn) not model(x)
            loss = criterion(model(xc, xn), y)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss: float = 0.0
        with torch.no_grad():
            for xc, xn, y in val_loader:
                v_loss += criterion(model(xc.to(device), xn.to(device)), y.to(device)).item()

        val_loss = v_loss / len(val_loader)
        # # Early stopping logic
        # if val_loss < best_val:
        #     best_val = val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        # if epochs_no_improve >= patience:
        #     break

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

        # Optuna pruning (highly recommended) - stops bad trials automatically
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return val_loss
    
