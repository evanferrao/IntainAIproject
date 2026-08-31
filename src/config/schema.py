"""Pydantic data validation and canonical schema models."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class CanonicalStaticLoan(BaseModel):
    loan_id: str
    origination_month: str
    original_balance: float = Field(..., gt=0)
    funded_balance: float = Field(..., gt=0)
    interest_rate: float = Field(..., gt=0, le=100.0)
    original_term: int = Field(..., gt=0)
    installment: float = Field(..., gt=0)
    credit_score: float = Field(..., ge=300, le=850)
    credit_score_band: str
    dti: float = Field(..., ge=0.0)
    dti_band: str
    annual_income: float = Field(..., gt=0)
    employment_length: int = Field(..., ge=0, le=10)
    home_ownership: str
    loan_purpose: str
    state: str
    revolving_utilization: float = Field(..., ge=0.0)
    delinquencies_2yrs: int = Field(..., ge=0)
    inquiries_6m: int = Field(..., ge=0)
    total_accounts: int = Field(..., ge=1)
    servicer_name: str
    document_status: str


class CanonicalMonthlyPerformance(BaseModel):
    loan_id: str
    reporting_month: str
    month_index: int = Field(..., ge=0)
    loan_age_months: int = Field(..., ge=0)
    remaining_term_months: int = Field(..., ge=0)
    current_balance: float = Field(..., ge=0.0)
    current_status: str
    days_past_due: int = Field(..., ge=0)
    modification_flag: int = Field(..., ge=0, le=1)
    prepayment_flag: int = Field(..., ge=0, le=1)
    default_flag: int = Field(..., ge=0, le=1)
    next_state: Optional[str] = None
    next_3m_delinquency_flag: Optional[int] = None
    next_12m_default_flag: Optional[int] = None
    next_12m_prepayment_flag: Optional[int] = None


class ServicerUpdateRecord(BaseModel):
    update_id: str
    loan_id: str
    servicer_name: str
    reported_status: str
    reported_balance: float
    reported_dpd: int
    document_status: str
    last_updated_at: str
    source_system: str


class SubmissionRecord(BaseModel):
    loan_id: str
    delinquency_probability: float = Field(..., ge=0.0, le=1.0)
    default_probability: float = Field(..., ge=0.0, le=1.0)
    prepayment_probability: float = Field(..., ge=0.0, le=1.0)
    next_state: str
    exception_probability: float = Field(..., ge=0.0, le=1.0)
    exception_type: str
    anomaly_score: float = Field(..., ge=0.0, le=100.0)
    top_drivers: str
    reviewer_action: str
    confidence: Literal["HIGH", "MODERATE", "LOW"]
