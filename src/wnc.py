import pandas as pd
import importlib

from . import (
    list_creditQ,
    list_credit_nonQ,
    list_equity,
    list_commodity,
    list_rates,
    list_fx,
    simm_tenor_list

)

# Version selection support
_current_version = "2.7"
_params = None

def set_simm_version(version: str):
    """
    Set the SIMM version for parameter loading.

    Args:
        version: "2.3", "2.4", "2.5", "2.6", or "2.7"
    """
    global _current_version, _params
    valid_versions = ["2.3", "2.4", "2.5", "2.6", "2.7"]
    if version not in valid_versions:
        raise ValueError(f"Invalid SIMM version: {version}. Valid: {valid_versions}")
    _current_version = version
    _params = None  # Force reload on next access
    _load_params()

def get_simm_version() -> str:
    """Return the currently loaded SIMM version."""
    return _current_version

def _load_params():
    """Load parameters for the current version."""
    global _params
    if _params is not None:
        return

    version_map = {
        "2.3": "Weights_and_Corr.v2_3",
        "2.4": "Weights_and_Corr.v2_4",
        "2.5": "Weights_and_Corr.v2_5",
        "2.6": "Weights_and_Corr.v2_6",
        "2.7": "Weights_and_Corr.v2_7",
    }
    module = importlib.import_module(version_map[_current_version])
    _params = module

# Load default version on module import
_load_params()

# Accessor functions to get parameters from loaded module
def _get_param(name):
    """Get a parameter from the loaded version module."""
    _load_params()
    return getattr(_params, name)


def RW(risk_class,bucket):
    if risk_class in list_creditQ:
        return _get_param('creditQ_rw')[bucket]

    elif risk_class in list_credit_nonQ:
        return _get_param('creiditNonQ_rw')[bucket]

    elif risk_class in list_equity:
        return _get_param('equity_rw')[bucket]

    elif risk_class in list_commodity:
        return _get_param('commodity_rw')[bucket]

def rho(risk_class,index1=None,index2=None,bucket=None):

    if risk_class in list_rates:
        return pd.DataFrame(
            _get_param('ir_corr'),
            columns=simm_tenor_list,
            index=simm_tenor_list
        )[index1][index2]

    elif risk_class in list_creditQ:
        creditQ_corr = _get_param('creditQ_corr')

        if risk_class == 'Risk_BaseCorr':
            return float(creditQ_corr[3])

        elif (index1 == 'Res') or (index2 == 'Res'):
            rho = creditQ_corr[2]
        elif index1 == index2:
            rho = creditQ_corr[0]
        else:
            rho = creditQ_corr[1]
        return float(rho)

    elif risk_class in list_credit_nonQ:
        creditNonQ_corr = _get_param('creditNonQ_corr')
        if (index1 == 'Res') or (index2 == 'Res'):
            rho = creditNonQ_corr[2]
        elif index1 == index2:
            rho = creditNonQ_corr[0]
        else:
            rho = creditNonQ_corr[1]
        return rho

    elif risk_class in list_equity:
        return _get_param('equity_corr')[bucket]

    elif risk_class in list_commodity:
        return _get_param('commodity_corr')[bucket]

def gamma(risk_class,bucket1=None,bucket2=None):

    if risk_class in list_creditQ:
        bucket_list = [str(i) for i in range(1,13)]
        return pd.DataFrame(
            _get_param('creditQ_corr_non_res'),
            columns=bucket_list,
            index=bucket_list
        )[bucket1][bucket2]

    elif risk_class in list_credit_nonQ:
        return _get_param('cr_gamma_diff_ccy')

    elif risk_class in list_equity:
        bucket_list = [str(i) for i in range(1,13)]
        return pd.DataFrame(
            _get_param('equity_corr_non_res'),
            columns=bucket_list,
            index=bucket_list
        )[bucket1][bucket2]

    elif risk_class in list_commodity:
        bucket_list = [str(i) for i in range(1,18)]
        return pd.DataFrame(
            _get_param('commodity_corr_non_res'),
            columns=bucket_list,
            index=bucket_list
        )[bucket1][bucket2]

def T(risk_class,type,currency=None,bucket=None):
    ir_delta_CT = _get_param('ir_delta_CT')
    credit_delta_CT = _get_param('credit_delta_CT')
    equity_delta_CT = _get_param('equity_delta_CT')
    commodity_delta_CT = _get_param('commodity_delta_CT')
    fx_category1 = _get_param('fx_category1')
    fx_category2 = _get_param('fx_category2')
    fx_delta_CT = _get_param('fx_delta_CT')
    ir_vega_CT = _get_param('ir_vega_CT')
    credit_vega_CT = _get_param('credit_vega_CT')
    equity_vega_CT = _get_param('equity_vega_CT')
    commodity_vega_CT = _get_param('commodity_vega_CT')
    fx_vega_CT = _get_param('fx_vega_CT')

    if type == 'Delta':
        if risk_class == 'Rates':
            try:
                T = ir_delta_CT[currency]
            except KeyError:
                T = ir_delta_CT['Others']

        elif risk_class in list_creditQ:
            T = credit_delta_CT['Qualifying'][bucket]

        elif risk_class in list_credit_nonQ:
            T = credit_delta_CT['Non-Qualifying'][bucket]

        elif risk_class in list_equity:
            T = equity_delta_CT[bucket]

        elif risk_class in list_commodity:
            T = commodity_delta_CT[bucket]

        elif risk_class in list_fx:
            if currency in fx_category1:
                T = fx_delta_CT['Category1']
            elif currency in fx_category2:
                T = fx_delta_CT['Category2']
            else:
                T = fx_delta_CT['Others']

    elif type == 'Vega':
        if risk_class == 'Rates':
            try:
                T = ir_vega_CT[currency]
            except KeyError:
                T = ir_vega_CT['Others']

        elif risk_class in list_creditQ:
            T = credit_vega_CT['Qualifying']

        elif risk_class in list_credit_nonQ:
            T = credit_vega_CT['Non-Qualifying']

        elif risk_class in list_equity:
            T = equity_vega_CT[bucket]

        elif risk_class in list_commodity:
            T = commodity_vega_CT[bucket]

        elif risk_class in list_fx:
            currency1 = currency[0:3]
            currency2 = currency[3:6]

            if (currency1 in fx_category1) and (currency2 in fx_category1):
                T = fx_vega_CT['Category1-Category1']

            elif ((currency1 in fx_category1) and (currency2 in fx_category2)) or ((currency1 in fx_category2) and (currency2 in fx_category1)):
                T = fx_vega_CT['Category1-Category2']

            elif ((currency1 in fx_category1) and (currency2 not in fx_category1+fx_category2)) or ((currency1 not in fx_category1+fx_category2) and (currency2 in fx_category1)):
                T = fx_vega_CT['Category1-Category3']

            elif (currency1 in fx_category2) and (currency2 in fx_category2):
                T = fx_vega_CT['Category2-Category2']

            elif ((currency1 in fx_category2) and (currency2 not in fx_category1+fx_category2)) or ((currency1 not in fx_category1+fx_category2) and (currency2 in fx_category2)):
                T = fx_vega_CT['Category2-Category3']

            elif (currency1 not in fx_category1+fx_category2) and (currency2 not in fx_category1+fx_category2):
                T = fx_vega_CT['Category3-Category3']

    return T * 1000000

def psi(risk_class1,risk_class2):
    return pd.DataFrame(
        _get_param('corr_params'),
        columns = ['Rates','CreditQ','CreditNonQ','Equity','Commodity','FX'],
        index   = ['Rates','CreditQ','CreditNonQ','Equity','Commodity','FX']
    )[risk_class1][risk_class2]


# Expose commonly used parameters via accessor functions
def get_ir_weights(volatility='regular'):
    """Get IR risk weights for given volatility category."""
    if volatility == 'regular':
        return _get_param('reg_vol_rw')
    elif volatility == 'low':
        return _get_param('low_vol_rw')
    elif volatility == 'high':
        return _get_param('high_vol_rw')

def get_ir_params():
    """Get IR-specific parameters."""
    return {
        'inflation_rw': _get_param('inflation_rw'),
        'ccy_basis_swap_spread_rw': _get_param('ccy_basis_swap_spread_rw'),
        'sub_curves_corr': _get_param('sub_curves_corr'),
        'inflation_corr': _get_param('inflation_corr'),
        'ccy_basis_spread_corr': _get_param('ccy_basis_spread_corr'),
        'ir_gamma_diff_ccy': _get_param('ir_gamma_diff_ccy'),
        'reg_vol_ccy_bucket': _get_param('reg_vol_ccy_bucket'),
        'low_vol_ccy_bucket': _get_param('low_vol_ccy_bucket'),
    }

def get_fx_params():
    """Get FX-specific parameters."""
    return {
        'fx_rw': _get_param('fx_rw'),
        'high_vol_currency_group': _get_param('high_vol_currency_group'),
        'fx_reg_vol_corr': _get_param('fx_reg_vol_corr'),
        'fx_high_vol_corr': _get_param('fx_high_vol_corr'),
    }


# =============================================================================
# Backward-compatible module-level attribute access
# These allow existing code to use wnc.reg_vol_ccy_bucket etc.
# =============================================================================

class _ParamProxy:
    """Proxy object that redirects attribute access to _get_param."""
    def __init__(self, param_name):
        self._param_name = param_name

    def __getitem__(self, key):
        return _get_param(self._param_name)[key]

    def __contains__(self, item):
        return item in _get_param(self._param_name)

    def __iter__(self):
        return iter(_get_param(self._param_name))

    def __repr__(self):
        return repr(_get_param(self._param_name))


class _ScalarProxy:
    """Proxy for scalar parameters."""
    def __init__(self, param_name):
        self._param_name = param_name

    def _value(self):
        return _get_param(self._param_name)

    def __float__(self):
        return float(self._value())

    def __int__(self):
        return int(self._value())

    def __repr__(self):
        return repr(self._value())

    def __eq__(self, other):
        return self._value() == other

    def __mul__(self, other):
        return self._value() * other

    def __rmul__(self, other):
        return other * self._value()

    def __add__(self, other):
        return self._value() + other

    def __radd__(self, other):
        return other + self._value()


# Use Python module __getattr__ for dynamic attribute access
def __getattr__(name):
    """Allow accessing parameters as module attributes."""
    # List of all known parameters
    _known_params = [
        'reg_vol_ccy_bucket', 'low_vol_ccy_bucket',
        'reg_vol_rw', 'low_vol_rw', 'high_vol_rw',
        'inflation_rw', 'ccy_basis_swap_spread_rw',
        'ir_hvr', 'ir_vrw', 'ir_corr',
        'sub_curves_corr', 'inflation_corr', 'ccy_basis_spread_corr',
        'ir_gamma_diff_ccy',
        'creditQ_rw', 'creditQ_vrw', 'base_corr_weight', 'creditQ_corr', 'creditQ_corr_non_res',
        'creiditNonQ_rw', 'creditNonQ_vrw', 'creditNonQ_corr', 'cr_gamma_diff_ccy',
        'equity_rw', 'equity_hvr', 'equity_vrw', 'equity_vrw_bucket_12',
        'equity_corr', 'equity_corr_non_res',
        'commodity_rw', 'commodity_hvr', 'commodity_vrw',
        'commodity_corr', 'commodity_corr_non_res',
        'high_vol_currency_group',
        'fx_rw', 'fx_hvr', 'fx_vrw',
        'fx_reg_vol_corr', 'fx_high_vol_corr', 'fx_vega_corr',
        'ir_delta_CT', 'credit_delta_CT', 'equity_delta_CT', 'commodity_delta_CT',
        'fx_category1', 'fx_category2', 'fx_delta_CT',
        'ir_vega_CT', 'credit_vega_CT', 'equity_vega_CT', 'commodity_vega_CT', 'fx_vega_CT',
        'corr_params',
    ]

    if name in _known_params:
        return _get_param(name)

    # Handle special case for FX_Corr which is used in margin_risk_class.py
    # This is a legacy parameter that should use fx_reg_vol_corr/fx_high_vol_corr
    # Index 4 appears to be used for Regular-Regular correlation (0.5 in v2.5)
    if name == 'FX_Corr':
        # Return a list-like object that provides FX correlation values
        # Index 4 = Regular-Regular correlation for regular calc currency
        fx_corr = _get_param('fx_reg_vol_corr')
        return [0, 0, 0, 0, fx_corr['Regular']['Regular']]

    raise AttributeError(f"module 'wnc' has no attribute '{name}'")
