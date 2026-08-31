"""Markov Monthly State Transition and Multi-Period Survival Projections."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config.settings import VALID_STATUSES, TERMINAL_STATUSES, RANDOM_SEED
from src.utils.logger import logger


class MarkovTransitionModel:
    """Estimates empirical state transition matrices and computes multi-period roll-forward projections."""

    def __init__(self, states: Optional[List[str]] = None, seed: int = RANDOM_SEED):
        self.states = states or VALID_STATUSES
        self.seed = seed
        self.terminal_states = [s for s in self.states if s in TERMINAL_STATUSES]
        self.global_transition_matrix: Optional[pd.DataFrame] = None
        self.segmented_transition_matrices: Dict[str, Dict[str, pd.DataFrame]] = {}

    def fit(
        self,
        panel_df: pd.DataFrame,
        segment_cols: Optional[List[str]] = None,
        state_col: str = "current_status",
        next_state_col: str = "next_state"
    ) -> "MarkovTransitionModel":
        """Calculates empirical single-month transition probability matrices from panel observations."""
        logger.info(f"Fitting Markov Transition Model on {len(panel_df):,} monthly transitions...")
        df = panel_df.dropna(subset=[state_col, next_state_col]).copy()

        # 1. Global Transition Matrix
        self.global_transition_matrix = self._compute_transition_matrix(df, state_col, next_state_col)

        # 2. Segmented Transition Matrices (e.g. by credit_score_band, dti_band, etc.)
        if segment_cols:
            for seg_col in segment_cols:
                if seg_col in df.columns:
                    self.segmented_transition_matrices[seg_col] = {}
                    for seg_val, group in df.groupby(seg_col):
                        if len(group) >= 50:
                            self.segmented_transition_matrices[seg_col][str(seg_val)] = self._compute_transition_matrix(
                                group, state_col, next_state_col
                            )

        logger.info("Fitted Global and Segmented Transition Matrices.")
        return self

    def _compute_transition_matrix(
        self,
        df: pd.DataFrame,
        state_col: str,
        next_state_col: str
    ) -> pd.DataFrame:
        """Computes normalized row-stochastic transition matrix from pairs of consecutive states."""
        # Cross-tabulation
        ct = pd.crosstab(
            df[state_col],
            df[next_state_col],
            dropna=False
        )

        # Ensure all canonical states exist in index and columns
        for s in self.states:
            if s not in ct.index:
                ct.loc[s] = 0
            if s not in ct.columns:
                ct[s] = 0

        ct = ct.reindex(index=self.states, columns=self.states, fill_value=0)

        # Row-normalize to get probabilities P(S_{t+1} | S_t)
        row_sums = ct.sum(axis=1)
        prob_matrix = ct.div(row_sums.replace(0, 1), axis=0)

        # Enforce terminal state absorption: P(DEFAULT -> DEFAULT) = 1.0, P(PREPAID -> PREPAID) = 1.0, etc.
        for ts in self.terminal_states:
            if ts in prob_matrix.index:
                prob_matrix.loc[ts, :] = 0.0
                prob_matrix.loc[ts, ts] = 1.0

        # Handle rows with 0 observations by assigning identity
        for s in self.states:
            if row_sums[s] == 0:
                prob_matrix.loc[s, :] = 0.0
                prob_matrix.loc[s, s] = 1.0

        return prob_matrix.round(4)

    def project_multi_period(
        self,
        initial_state_dist: Optional[Dict[str, float]] = None,
        horizon_months: int = 36,
        segment_col: Optional[str] = None,
        segment_val: Optional[str] = None
    ) -> pd.DataFrame:
        """Projects portfolio state probability distributions over time using Markov matrix powers P^h."""
        if self.global_transition_matrix is None:
            raise RuntimeError("Model must be fitted before running multi-period projections.")

        # Choose transition matrix (Segmented or Global)
        if segment_col and segment_val and segment_col in self.segmented_transition_matrices:
            T = self.segmented_transition_matrices[segment_col].get(segment_val, self.global_transition_matrix).values
        else:
            T = self.global_transition_matrix.values

        # Initial distribution vector (Default: 100% CURRENT)
        if initial_state_dist is None:
            v0 = np.zeros(len(self.states))
            if "CURRENT" in self.states:
                v0[self.states.index("CURRENT")] = 1.0
            else:
                v0[0] = 1.0
        else:
            v0 = np.array([initial_state_dist.get(s, 0.0) for s in self.states])
            v0 = v0 / max(1e-8, v0.sum())

        projections = []
        current_v = v0.copy()

        default_idx = self.states.index("DEFAULT") if "DEFAULT" in self.states else -1
        prepay_idx = self.states.index("PREPAID") if "PREPAID" in self.states else -1

        for h in range(horizon_months + 1):
            row_dict = {"month": h}
            for idx, s in enumerate(self.states):
                row_dict[s] = round(float(current_v[idx]), 4)

            # Cumulative absorption
            row_dict["cumulative_default"] = round(float(current_v[default_idx]), 4) if default_idx >= 0 else 0.0
            row_dict["cumulative_prepayment"] = round(float(current_v[prepay_idx]), 4) if prepay_idx >= 0 else 0.0

            projections.append(row_dict)

            # Roll forward by one period: v_{h+1} = v_h * T
            current_v = np.dot(current_v, T)

        proj_df = pd.DataFrame(projections)
        return proj_df

    def to_dict(self) -> Dict[str, Any]:
        """Serializes transition matrices to dictionary."""
        return {
            "states": self.states,
            "terminal_states": self.terminal_states,
            "global_matrix": self.global_transition_matrix.to_dict() if self.global_transition_matrix is not None else {},
            "segmented_matrices": {
                scol: {sval: mat.to_dict() for sval, mat in seg_dict.items()}
                for scol, seg_dict in self.segmented_transition_matrices.items()
            }
        }
