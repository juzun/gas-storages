import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GasPriceSimulations:
    def __init__(self, n_paths: int = 1000):
        self.data_path = Path("../data/")
        self.n_paths = n_paths
        self.index: pd.DatetimeIndex | None = None

    def get_simulations(self) -> np.ndarray:
        p_series = self.get_gas_fwd()
        p = p_series.values

        n_steps = len(p_series)
        mu = np.zeros(n_steps)
        dt = 1.0 / 365.0
        kappa = 3.0
        sigma = 6.0

        for t in range(n_steps - 1):
            mu[t] = (p[t + 1] - (1 - kappa * dt) * p[t]) / (kappa * dt)

        # fix last price
        mu[-1] = p[-1]

        paths = self.simulate_gas_spot(p=p, mu=mu, kappa=kappa, sigma=sigma, dt=dt, n_paths=1000)

        return paths

    def get_gas_fwd(self) -> pd.Series:
        gas_fwd = pd.read_csv(self.data_path / "gas_fwd.csv", parse_dates=True, index_col="date")
        gas_fwd.index.name = None
        gas_fwd = gas_fwd.reindex(
            pd.date_range(
                gas_fwd.index.min(), gas_fwd.index.max() + pd.offsets.MonthEnd(1), freq="D"
            ),
            method="ffill",
        )
        self.index = gas_fwd.index
        return gas_fwd["price"]

    @staticmethod
    def simulate_gas_spot(
        p: np.ndarray,
        mu: np.ndarray,
        kappa: float,
        sigma: float,
        dt: float,
        n_paths: int = 1000,
    ) -> np.ndarray:
        n = len(p)
        paths = np.zeros((n_paths, n))
        paths[:, 0] = p[0]

        np.random.seed(42)
        z_dist = np.random.normal(size=(n_paths, n - 1))

        for t in range(n - 1):
            paths[:, t + 1] = (
                paths[:, t]
                + kappa * (mu[t] - paths[:, t]) * dt
                + sigma * np.sqrt(dt) * z_dist[:, t]
            )

        return paths
    