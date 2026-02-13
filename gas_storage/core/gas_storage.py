import datetime as dt
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
from pyomo.contrib import appsi
import pyomo.environ as pyo


class GasStorage:
    """Gas storage optimization model using mixed-integer linear programming.
    
    This class implements a comprehensive gas storage optimization framework that
    maximizes profit from buy/sell operations while respecting physical constraints
    such as capacity limits, injection/withdrawal curves, and seasonal restrictions.
    """
    
    def __init__(self, date_start: dt.date, date_end: dt.date) -> None:
        """Initialize a gas storage optimization instance.
        
        Creates a daily-indexed dataframe for the specified date range and initializes
        all storage parameters, model components, and result containers.
        
        Args:
            date_start: Start date of the optimization period (inclusive).
            date_to: End date of the optimization period (inclusive).
        """
        # Create daily-indexed dataframe for the entire optimization period
        self.attr = pd.DataFrame(index=pd.date_range(date_start, date_end, freq="D"))

        self.z0: int = 0
        self.optimization_time_limit: int = 3600
        self.empty_storage: bool = False
        self.empty_on_dates: List[dt.date] = []
        self.bsd_state_to_date: Dict[int, float] = {}
        self.injection_season: List[int] = []
        self.injection_idx: List[int] = []
        self.withdrawal_idx: List[int] = []
        self.inj_curve_daily: Dict[Tuple[dt.date, int, str], float] = {}
        self.wit_curve_daily: Dict[Tuple[dt.date, int, str], float] = {}

        self.mdl: Optional[pyo.ConcreteModel] = None
        self.slvr: Optional[Union[appsi.solvers.Cplex, appsi.solvers.Highs]] = None
        self.objective: Optional[float] = None
        self.results: Optional[
            Union[appsi.solvers.highs.HighsResults, appsi.solvers.cplex.CplexResults]
        ] = None

        self.best_objective_bound: Optional[float] = None
        self.best_feasible: Optional[float] = None
        self.gap: Optional[float] = None

    def _transform_curve(self, curve_type: Literal["inj_curve", "wit_curve"]) -> None:
        """
        Converts curve data from dataframe storage format (2D arrays per date) into
        indexed dictionaries suitable for Pyomo parameter initialization. Each curve
        consists of piecewise linear segments with lower/upper bounds and portions.
        
        Args:
            curve_type: Type of curve to transform, either "inj_curve" for injection
                       or "wit_curve" for withdrawal curves.
        """
        curve_dict = self.attr[curve_type].to_dict()
        labels = ["lower", "upper", "portion"]
        
        if curve_type == "inj_curve":
            for date, curve_data in curve_dict.items():
                for i in range(len(self.injection_idx)):
                    for idx, label in enumerate(labels):
                        self.inj_curve_daily[date, self.injection_idx[i], label] = curve_data[idx, i]
        else:
            for date, curve_data in curve_dict.items():
                for i in range(len(self.withdrawal_idx)):
                    for idx, label in enumerate(labels):
                        self.wit_curve_daily[date, self.withdrawal_idx[i], label] = curve_data[idx, i]

    def load_attribute(
        self,
        attr_name: str,
        value: np.ndarray | int,
        date_from: dt.date,
        date_to: dt.date,
    ) -> None:
        """Load storage attribute values for a specific date range.
        
        Stores attribute values (capacity, rates, prices, curves) in the attributes
        dataframe for the specified date range. Handles both scalar values and complex
        curve arrays, with automatic processing of dependent attributes.
        
        Args:
            attr_name: Name of attribute to load (e.g., "wgv", "ir", "wr", "price",
                      "inj_curve", "wit_curve").
            value: Attribute value(s). Scalar for simple attributes (capacity, rates),
                  numpy array for curves.
            date_from: Start date of the period (inclusive).
            date_to: End date of the period (inclusive).

        Note:
            When loading curve data, automatically extracts segment indices and
            transforms curves into model-ready format.
        """
        # Ensure column exists in dataframe
        if attr_name not in self.attr:
            self.attr[attr_name] = None
        
        # Select rows within the specified date range
        selected_rows = (self.attr.index >= pd.to_datetime(date_from)) & (
            self.attr.index <= pd.to_datetime(date_to)
        )
        # Assign value to all selected dates
        self.attr.loc[selected_rows, attr_name] = pd.Series(
            [value] * selected_rows.sum(), index=self.attr.index[selected_rows]
        )

        # Handle special attributes with automatic dependent processing
        if attr_name == "inj_curve":
            # Extract number of curve segments and create segment indices
            num_segments = self.attr["inj_curve"].iloc[0].shape[1]
            self.injection_idx = list(range(1, num_segments + 1))
            # Transform curve into model-compatible dictionary format
            self._transform_curve(attr_name)
        elif attr_name == "wit_curve":
            # Extract number of curve segments and create segment indices
            num_segments = self.attr["wit_curve"].iloc[0].shape[1]
            self.withdrawal_idx = list(range(1, num_segments + 1))
            # Transform curve into model-compatible dictionary format
            self._transform_curve(attr_name)

    def _mdl_initialize_sets(self) -> None:
        """
        Creates index sets for all model dimensions including time periods, curve
        segments, and special date constraints. These sets define the valid indices
        for variables, parameters, and constraints.
        """
        self.mdl.i = pyo.Set(initialize=self.attr.index)
        self.mdl.j = pyo.Set(initialize=self.injection_idx)
        self.mdl.k = pyo.Set(initialize=self.withdrawal_idx)
        # Months with minimum state requirements
        self.mdl.bsd_months = pyo.Set(initialize=list(self.bsd_state_to_date.keys()))
        # Season-restricted day sets — curve binaries only created for active season
        self.mdl.inj_days = pyo.Set(initialize=[
            d for d in self.attr.index if d.month in self.injection_season
        ])
        self.mdl.wit_days = pyo.Set(initialize=[
            d for d in self.attr.index if d.month not in self.injection_season
        ])
        # # Curve value types: bounds and portion for piecewise linear curves
        # self.mdl.curve_value_type = pyo.Set(initialize=["lower", "upper", "portion"])

    def _mdl_initialize_params(self) -> None:
        """
        Creates model parameters. Curve data (inj/wit) is accessed directly from
        self.inj_curve_daily / self.wit_curve_daily dicts in constraints, avoiding
        the overhead of creating large indexed Pyomo Param objects.
        """
        self.mdl.p = pyo.Param(self.mdl.i, initialize=self.attr["price"].to_dict())
        self.mdl.wgv = pyo.Param(self.mdl.i, initialize=self.attr["wgv"].to_dict())
        self.mdl.ir = pyo.Param(self.mdl.i, initialize=self.attr["ir"].to_dict())
        self.mdl.wr = pyo.Param(self.mdl.i, initialize=self.attr["wr"].to_dict())
        self.mdl.bsd_state_to_date = pyo.Param(
            self.mdl.bsd_months, initialize=self.bsd_state_to_date
        )

        # # Big-M constant for logical constraints [MWh]
        # self.mdl.m_const = pyo.Param(self.mdl.i, initialize=self.attr["m_const"].to_dict())
        # # Injection curve: piecewise linear approximation indexed by (date, segment, type)
        # self.mdl.tab_inj = pyo.Param(
        #     self.mdl.i,
        #     self.mdl.j,
        #     self.mdl.curve_value_type,
        #     initialize=self.inj_curve_daily,
        # )
        # # Withdrawal curve: piecewise linear approximation indexed by (date, segment, type)
        # self.mdl.tab_wit = pyo.Param(
        #     self.mdl.i,
        #     self.mdl.k,
        #     self.mdl.curve_value_type,
        #     initialize=self.wit_curve_daily,
        # )

    def _mdl_initialize_vars(self) -> None:
        """
        Creates decision variables. Binary curve-segment indicators (t_inj, t_wit)
        are only created for the active season, roughly halving the binary count.
        Variables l_inj/u_inj/l_wit/u_wit from the old triple-binary formulation
        are eliminated entirely.
        """
        # Injection volume by date [MWh/day]
        self.mdl.x = pyo.Var(
            self.mdl.i, domain=pyo.NonNegativeReals, initialize=0, name="x"
        )
        # Withdrawal volume by date [MWh/day]
        self.mdl.y = pyo.Var(
            self.mdl.i, domain=pyo.NonNegativeReals, initialize=0, name="y"
        )
        # Storage state (inventory) by date [MWh]
        self.mdl.z = pyo.Var(
            self.mdl.i, domain=pyo.NonNegativeReals, initialize=0, name="z"
        )

        # Binary variables for injection curve piecewise approximation
        # t_inj[i,j] = 1 if storage state is in segment j on date i
        # Only created for injection-season days
        self.mdl.t_inj = pyo.Var(
            self.mdl.inj_days, self.mdl.j, domain=pyo.Binary, initialize=0, name="t_inj"
        )
        # # l_inj[i,j] = 1 if state is at or above lower bound of segment j
        # self.mdl.l_inj = pyo.Var(
        #     self.mdl.i, self.mdl.j, domain=pyo.Binary, initialize=0, name="l_inj"
        # )
        # # u_inj[i,j] = 1 if state is at or below upper bound of segment j
        # self.mdl.u_inj = pyo.Var(
        #     self.mdl.i, self.mdl.j, domain=pyo.Binary, initialize=0, name="u_inj"
        # )

        # Binary variables for withdrawal curve piecewise approximation
        # t_wit[i,k] = 1 if storage state is in segment k on date i
        # Only created for withdrawal-season days
        self.mdl.t_wit = pyo.Var(
            self.mdl.wit_days, self.mdl.k, domain=pyo.Binary, initialize=0, name="t_wit"
        )
        # # l_wit[i,k] = 1 if state is at or above lower bound of segment k
        # self.mdl.l_wit = pyo.Var(
        #     self.mdl.i, self.mdl.k, domain=pyo.Binary, initialize=0, name="l_wit"
        # )
        # # u_wit[i,k] = 1 if state is at or below upper bound of segment k
        # self.mdl.u_wit = pyo.Var(
        #     self.mdl.i, self.mdl.k, domain=pyo.Binary, initialize=0, name="u_wit"
        # )

    def _mdl_def_constraints(self) -> None:
        """
        Defines all model constraints.

        Reformulation vs. original triple-binary approach:
        ──────────────────────────────────────────────────
        OLD — For each curve segment j on *every* day i, three binaries
        (l[i,j], u[i,j], t[i,j]) with 6·J + 1 big-M constraints per day
        detect which segment the fill level z[i] falls into.

        NEW — Only t[i,j] is kept. Two direct big-M constraints per segment
        enforce segment membership when t[i,j] = 1:

            z[i] ≥ lower_j · WGV - WGV · (1 - t[i,j])     (1)
            z[i] ≤ upper_j · WGV + WGV · (1 - t[i,j])     (2)
            Σ_j t[i,j] = 1                                  (3)

        • For that active j: (1),(2) tighten to lower_j·WGV ≤ z ≤ upper_j·WGV,
          so the solver must pick the segment that truly contains z[i].
        • For inactive j (t=0): (1),(2) relax to z ≥ (lower_j-1)·WGV and
          z ≤ (upper_j+1)·WGV — satisfied by any z ∈ [0, WGV]. This is
          intentional: big-M constraints must be non-restrictive when inactive.
        • The rate constraint x[i] ≤ ir[i] · Σ_j portion_j · t[i,j] reduces
          to x[i] ≤ ir[i] · portion_active since only one t[i,j] = 1.
        """
        # # Global mass balance: total withdrawal cannot exceed initial storage plus total injection
        # self.mdl.constr_balance = pyo.Constraint(
        #     expr=sum(self.mdl.y[i] for i in self.mdl.i)
        #     <= self.z0 + sum(self.mdl.x[i] for i in self.mdl.i)
        # )

        # Empty storage constraints: enforce zero inventory on specific dates
        self.mdl.constr_empty_storage = pyo.ConstraintList()
        if self.empty_storage:
            for date in self.empty_on_dates:
                # Only add constraint if date is within optimization period
                if (date >= self.attr.index[0]) and (date <= self.attr.index[-1]):
                    self.mdl.constr_empty_storage.add(self.mdl.z[date] == 0)

        # Capacity constraints: storage state cannot exceed working gas volume
        self.mdl.constr_capacity = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_capacity.add(self.mdl.z[i] <= self.mdl.wgv[i])

        # Gas balance constraints: track daily storage state evolution
        self.mdl.constr_gs = pyo.ConstraintList()
        for i in self.mdl.i:
            if i == self.attr.index[0]:
                # First day: state = initial level + injection - withdrawal
                self.mdl.constr_gs.add(
                    self.mdl.z[i] == self.z0 + self.mdl.x[i] - self.mdl.y[i]
                )
                continue
            # Subsequent days: state = previous state + injection - withdrawal
            self.mdl.constr_gs.add(
                self.mdl.z[i]
                == self.mdl.z[i - dt.timedelta(days=1)] + self.mdl.x[i] - self.mdl.y[i]
            )

        # Seasonal restrictions: restrict to injection or withdrawal in specific months
        self.mdl.constr_season = pyo.ConstraintList()
        for i in self.mdl.i:
            if i.month in self.injection_season:
                # Injection season: no withdrawal allowed
                self.mdl.constr_season.add(self.mdl.y[i] == 0)
            else:
                # Withdrawal season: no injection allowed
                self.mdl.constr_season.add(self.mdl.x[i] == 0)

        # Minimum state requirements: enforce minimum fill level on first day of specified months
        self.mdl.constr_state_to_date = pyo.ConstraintList()
        for i in self.mdl.i:
            for p in self.mdl.bsd_months:
                # Check if this is the first day of a constrained month
                if i.month == p and i.day == 1:
                    self.mdl.constr_state_to_date.add(
                        self.mdl.z[i] >= self.mdl.bsd_state_to_date[p] * self.mdl.wgv[i]
                    )

        # # Injection curve lower bound constraints (piecewise linear approximation)
        # self.mdl.constr_inj_low = pyo.ConstraintList()
        # for i in self.mdl.i:
        #     for j in self.mdl.j:
        #         # If l_inj[i,j]=1, then z[i] >= lower_bound[j] * capacity
        #         self.mdl.constr_inj_low.add(
        #             self.mdl.tab_inj[(i, j, "lower")] * self.mdl.wgv[i]
        #             <= self.mdl.z[i] + self.mdl.m_const[i] * (1 - self.mdl.l_inj[i, j])
        #         )
        #         # If l_inj[i,j]=0, constraint is inactive (big-M method)
        #         self.mdl.constr_inj_low.add(
        #             self.mdl.tab_inj[(i, j, "lower")] * self.mdl.wgv[i]
        #             >= self.mdl.z[i] - self.mdl.m_const[i] * self.mdl.l_inj[i, j]
        #         )                
        # # Injection curve upper bound constraints
        # self.mdl.constr_inj_upp = pyo.ConstraintList()
        # for i in self.mdl.i:
        #     for j in self.mdl.j:
        #         # If u_inj[i,j]=1, then z[i] <= upper_bound[j] * capacity
        #         self.mdl.constr_inj_upp.add(
        #             self.mdl.tab_inj[(i, j, "upper")] * self.mdl.wgv[i]
        #             >= self.mdl.z[i] - self.mdl.m_const[i] * (1 - self.mdl.u_inj[i, j])
        #         )
        #         # If u_inj[i,j]=0, constraint is inactive
        #         self.mdl.constr_inj_upp.add(
        #             self.mdl.tab_inj[(i, j, "upper")] * self.mdl.wgv[i]
        #             <= self.mdl.z[i] + self.mdl.m_const[i] * self.mdl.u_inj[i, j]
        #         )
        # # Injection curve segment selection constraints
        # self.mdl.constr_inj_t = pyo.ConstraintList()
        # for i in self.mdl.i:
        #     # Exactly one segment must be active on each day
        #     self.mdl.constr_inj_t.add(
        #         sum(self.mdl.t_inj[i, j] for j in self.mdl.j) == 1
        #     )
        #     for j in self.mdl.j:
        #         # t_inj[i,j]=1 if both l_inj[i,j]=1 and u_inj[i,j]=1
        #         self.mdl.constr_inj_t.add(
        #             self.mdl.u_inj[i, j]
        #             + self.mdl.l_inj[i, j]
        #             - 2 * self.mdl.t_inj[i, j]
        #             >= 0
        #         )
        #         self.mdl.constr_inj_t.add(
        #             self.mdl.u_inj[i, j]
        #             + self.mdl.l_inj[i, j]
        #             - 2 * self.mdl.t_inj[i, j]
        #             <= 1
        #         )                
        # # Injection rate constraints: limit based on active curve segment
        # self.mdl.constr_inj = pyo.ConstraintList()
        # for i in self.mdl.i:
        #     # Injection cannot exceed max rate * portion for active segment
        #     self.mdl.constr_inj.add(
        #         self.mdl.x[i]
        #         <= self.mdl.ir[i]
        #         * sum(
        #             self.inj_curve_daily[(i, j, "portion")] * self.mdl.t_inj[i, j]
        #             for j in self.mdl.j
        #         )
        #     )

        # Injection curve constraints (injection-season days only)
        self.mdl.constr_inj_t = pyo.ConstraintList()
        self.mdl.constr_inj = pyo.ConstraintList()
        for i in self.mdl.inj_days:
            # Exactly one segment active on each day
            self.mdl.constr_inj_t.add(
                sum(self.mdl.t_inj[i, j] for j in self.mdl.j) == 1
            )
            for j in self.mdl.j:
                lo = self.inj_curve_daily[(i, j, "lower")]
                hi = self.inj_curve_daily[(i, j, "upper")]
                # If t=1: z >= lo*WGV  |  If t=0: z >= (lo-1)*WGV (trivial)
                self.mdl.constr_inj_t.add(
                    self.mdl.z[i] >= lo * self.mdl.wgv[i]
                    - self.mdl.wgv[i] * (1 - self.mdl.t_inj[i, j])
                )
                # If t=1: z <= hi*WGV  |  If t=0: z <= (hi+1)*WGV (trivial)
                self.mdl.constr_inj_t.add(
                    self.mdl.z[i] <= hi * self.mdl.wgv[i]
                    + self.mdl.wgv[i] * (1 - self.mdl.t_inj[i, j])
                )
            # Rate limit: x <= ir * portion_of_active_segment
            self.mdl.constr_inj.add(
                self.mdl.x[i]
                <= self.mdl.ir[i]
                * sum(
                    self.inj_curve_daily[(i, j, "portion")] * self.mdl.t_inj[i, j]
                    for j in self.mdl.j
                )
            )

        # # Withdrawal curve lower bound constraints (analogous to injection)
        # self.mdl.constr_wit_low = pyo.ConstraintList()
        # for i in self.mdl.i:
        #     for k in self.mdl.k:
        #         # If l_wit[i,k]=1, then z[i] >= lower_bound[k] * capacity
        #         self.mdl.constr_wit_low.add(
        #             self.mdl.tab_wit[(i, k, "lower")] * self.mdl.wgv[i]
        #             <= self.mdl.z[i] + self.mdl.m_const[i] * (1 - self.mdl.l_wit[i, k])
        #         )
        #         self.mdl.constr_wit_low.add(
        #             self.mdl.tab_wit[(i, k, "lower")] * self.mdl.wgv[i]
        #             >= self.mdl.z[i] - self.mdl.m_const[i] * self.mdl.l_wit[i, k]
        #         )                
        # # Withdrawal curve upper bound constraints
        # self.mdl.constr_wit_upp = pyo.ConstraintList()
        # for i in self.mdl.i:
        #     for k in self.mdl.k:
        #         # If u_wit[i,k]=1, then z[i] <= upper_bound[k] * capacity
        #         self.mdl.constr_wit_upp.add(
        #             self.mdl.tab_wit[(i, k, "upper")] * self.mdl.wgv[i]
        #             >= self.mdl.z[i] - self.mdl.m_const[i] * (1 - self.mdl.u_wit[i, k])
        #         )
        #         self.mdl.constr_wit_upp.add(
        #             self.mdl.tab_wit[(i, k, "upper")] * self.mdl.wgv[i]
        #             <= self.mdl.z[i] + self.mdl.m_const[i] * self.mdl.u_wit[i, k]
        #         )
        # # Withdrawal curve segment selection constraints
        # self.mdl.constr_wit_t = pyo.ConstraintList()
        # for i in self.mdl.i:
        #     # Exactly one segment must be active on each day
        #     self.mdl.constr_wit_t.add(
        #         sum(self.mdl.t_wit[i, k] for k in self.mdl.k) == 1
        #     )
        #     for k in self.mdl.k:
        #         # t_wit[i,k]=1 iff both l_wit[i,k]=1 and u_wit[i,k]=1
        #         self.mdl.constr_wit_t.add(
        #             self.mdl.u_wit[i, k]
        #             + self.mdl.l_wit[i, k]
        #             - 2 * self.mdl.t_wit[i, k]
        #             >= 0
        #         )
        #         self.mdl.constr_wit_t.add(
        #             self.mdl.u_wit[i, k]
        #             + self.mdl.l_wit[i, k]
        #             - 2 * self.mdl.t_wit[i, k]
        #             <= 1
        #         )                
        # # Withdrawal rate constraints: limit based on active curve segment
        # self.mdl.constr_wit = pyo.ConstraintList()
        # for i in self.mdl.i:
        #     # Withdrawal cannot exceed max rate * portion for active segment
        #     self.mdl.constr_wit.add(
        #         self.mdl.y[i]
        #         <= self.mdl.wr[i]
        #         * sum(
        #             self.wit_curve_daily[(i, k, "portion")] * self.mdl.t_wit[i, k]
        #             for k in self.mdl.k
        #         )
        #     )

        # Withdrawal curve constraints (withdrawal-season days only)
        self.mdl.constr_wit_t = pyo.ConstraintList()
        self.mdl.constr_wit = pyo.ConstraintList()
        for i in self.mdl.wit_days:
            self.mdl.constr_wit_t.add(
                sum(self.mdl.t_wit[i, k] for k in self.mdl.k) == 1
            )
            for k in self.mdl.k:
                lo = self.wit_curve_daily[(i, k, "lower")]
                hi = self.wit_curve_daily[(i, k, "upper")]
                self.mdl.constr_wit.add(
                    self.mdl.z[i] >= lo * self.mdl.wgv[i]
                    - self.mdl.wgv[i] * (1 - self.mdl.t_wit[i, k])
                )
                self.mdl.constr_wit.add(
                    self.mdl.z[i] <= hi * self.mdl.wgv[i]
                    + self.mdl.wgv[i] * (1 - self.mdl.t_wit[i, k])
                )
            # Rate limit: y <= wr * portion_of_active_segment
            self.mdl.constr_wit.add(
                self.mdl.y[i]
                <= self.mdl.wr[i]
                * sum(
                    self.wit_curve_daily[(i, k, "portion")] * self.mdl.t_wit[i, k]
                    for k in self.mdl.k
                )
            )

    def create_model(self) -> None:
        """
        Initializes a complete MILP model by creating the Pyomo ConcreteModel and
        populating it with sets, parameters, variables, objective function, and
        constraints. The objective maximizes profit from gas trading operations.
        """
        self.mdl = pyo.ConcreteModel()
        self._mdl_initialize_sets()
        self._mdl_initialize_params()
        self._mdl_initialize_vars()
        self.mdl.objective = pyo.Objective(
            expr=(
                sum(self.mdl.y[i] * self.mdl.p[i] for i in self.mdl.i)
                - sum(self.mdl.x[i] * self.mdl.p[i] for i in self.mdl.i)
            ),
            sense=pyo.maximize,
        )
        self._mdl_def_constraints()

    def solve_model(
        self,
        solver_name: Literal["cplex", "highs", "scip"],
        gap: Optional[float] = None,
        stream_solver: bool = True,
        presolve_highs: Literal["off", "choose", "on"] = "choose",
        presolve_scip: Optional[int] = None,
    ) -> None:
        """
        Configures and runs the selected MILP solver with appropriate settings,
        then extracts and displays results if a valid solution is found.
        
        Args:
            solver_name: MILP solver to use ("cplex", "highs", or "scip").
            gap: Relative optimality gap tolerance (e.g., 0.01 for 1% gap).
                If None, solver default is used.
            stream_solver: If True, display solver output during optimization.
            presolve_highs: Presolve level for HiGHS solver.
            presolve_scip: Maximum presolve rounds for SCIP solver.
                If None, solver default is used.
        """
        if solver_name == "scip":
            self.slvr = pyo.SolverFactory("scip")
            # Use all available CPU threads
            self.slvr.options["lp/threads"] = psutil.cpu_count(logical=True)
            # Set time limit in seconds
            self.slvr.options["limits/time"] = self.optimization_time_limit
            if gap is not None:
                # Set relative optimality gap tolerance
                self.slvr.options["limits/gap"] = gap
            if presolve_scip is not None:
                # Limit number of presolve rounds
                self.slvr.options["presolving/maxrounds"] = presolve_scip

            # Solve model with SCIP
            self.results = self.slvr.solve(self.mdl, tee=stream_solver)
            self.termination_condition = self.results.solver.termination_condition
            
            if self._is_solution_valid():
                self._extract_values_from_model()
                self.objective = self.mdl.objective()
                self.best_feasible_objective = self.results.solver.primal_bound
                self.best_objective_bound = self.results.solver.dual_bound
                self._print_results()
        else:
            if solver_name == "cplex":
                self.slvr = appsi.solvers.Cplex()
                self.slvr.cplex_options = {"threads": psutil.cpu_count(logical=True)}
            if solver_name == "highs":
                self.slvr = appsi.solvers.Highs()
                self.slvr.highs_options = {
                    "threads": psutil.cpu_count(logical=True),
                    "presolve": presolve_highs,
                }

            self.slvr.config.time_limit = self.optimization_time_limit
            if gap is not None:
                self.slvr.config.mip_gap = gap
            self.slvr.config.stream_solver = stream_solver
            self.slvr.config.load_solution = False

            self.results = self.slvr.solve(self.mdl)
            self.termination_condition = self.results.termination_condition
            
            if self._is_solution_valid():
                self.results.solution_loader.load_vars()
                self._extract_values_from_model()
                self.objective = self.mdl.objective()
                self.best_feasible_objective = self.results.best_feasible_objective
                self.best_objective_bound = self.results.best_objective_bound
                self._print_results()

    def _extract_values_from_model(self) -> None:
        """Extract optimal solution values from the solved model.
        
        Retrieves decision variable values from the Pyomo model and stores them
        in instance dictionaries for further analysis and visualization. Also
        computes model statistics.
        
        Raises:
            AssertionError: If model has not been created (self.mdl is None).
        """
        assert self.mdl is not None
        # Compute and store model statistics (variable counts, constraint counts, etc.)
        self.mdl.compute_statistics()
        self.statistics = self.mdl.statistics

        # Extract decision variable values as dictionaries indexed by date
        self.res_injection = self.mdl.x.extract_values()  # Daily injection [MWh/day]
        self.res_withdrawal = self.mdl.y.extract_values()  # Daily withdrawal [MWh/day]
        self.res_gs_state = self.mdl.z.extract_values()  # Daily storage state [MWh]
        # Calculate net operations (positive = injection, negative = withdrawal)
        self.res_operations = {
            date: self.res_injection[date] - self.res_withdrawal[date]
            for date in self.attr.index
        }
        
        # Extract maximum possible operations based on active curve segments and rates
        ir = self.mdl.ir.extract_values()
        wr = self.mdl.wr.extract_values()
        t_inj = self.mdl.t_inj.extract_values()
        t_wit = self.mdl.t_wit.extract_values()
        self.max_operations = {}
        for date in self.attr.index:
            if date.month in self.injection_season:
                self.max_operations[date] = ir[date] * sum(
                    self.inj_curve_daily[(date, j, "portion")] * t_inj[date, j]
                    for j in list(self.mdl.j)
                )
            else:
                self.max_operations[date] = -wr[date] * sum(
                    self.wit_curve_daily[(date, k, "portion")] * t_wit[date, k]
                    for k in list(self.mdl.k)
                )

    def _is_solution_valid(self) -> bool:
        """
        Validates that the solver terminated successfully and found a feasible
        solution, either optimal or within time/gap limit with a valid bound.
        
        Returns:
            True if solution is valid and can be used, False otherwise.
        """
        has_feasible = self.results.solver.primal_bound is not None
        return (
            self.results.solver.status == pyo.SolverStatus.ok
            and (
                self.termination_condition == pyo.TerminationCondition.optimal
                or (has_feasible and self.termination_condition
                    in (pyo.TerminationCondition.maxTimeLimit,
                        pyo.TerminationCondition.other))
            )
        )

    def _print_results(self) -> None:
        """
        Displays key optimization metrics including termination status, objective
        value, bounds, and optimality gap. Calculates gap as relative difference
        between best bound and best solution.
        """
        # Calculate relative optimality gap
        self.gap = (
            (self.best_objective_bound - self.best_feasible_objective)
            / self.best_feasible_objective
            if self.best_feasible_objective > 0
            else None  # Avoid division by zero
        )
        # Display optimization results
        print(f"\nTermination condition: {self.termination_condition}")
        print(f"Best feasible objective: {self.best_feasible_objective}")
        print(f"Best objective bound: {self.best_objective_bound}")
        print(f"Gap: {self.gap}")
        print(f"Objective: {self.objective}\n")

    def graph(self) -> go.Figure:
        """Create interactive visualization of gas storage optimization results.
        
        Generates a Plotly figure showing storage operations and state over time,
        with dual y-axes for operations (MWh/day) and storage level (MWh).
        
        Returns:
            Plotly Figure object with three traces: maximum operations, actual
            operations, and storage state.
            
        Note:
            Requires solution to be available (must call solve_model first).
            Uses color scheme: orange for max operations, green for actual
            operations, and cyan for storage state.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.attr.index,
                y=list(self.max_operations.values()),
                name="Max. operations",
                line_color="#ffa600",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.attr.index,
                y=list(self.res_operations.values()),
                name="Operations",
                fill="tozeroy",
                line_color="#74d576",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.attr.index,
                y=list(self.res_gs_state.values()),
                name="GS state",
                fill="tozeroy",
                line_color="#34dbeb",
                yaxis="y2",
            )
        )
        fig = fig.update_layout(
            title=("Gas storage operations"),
            xaxis_title="Time",
            yaxis=dict(title="Operations [MWh/day]"),
            yaxis2=dict(
                title="GS state [MWh]",
                side="right",
                overlaying="y",
                tickfont=dict(color="#34dbeb"),
            ),
            legend=dict(orientation="v", x=1.06, xanchor="left", y=1),
        )
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor="grey")
        return fig
