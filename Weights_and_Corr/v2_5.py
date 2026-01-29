# ISDA SIMM v2.5 Parameters
# Extracted from AcadiaSoft simm-lib: https://github.com/AcadiaSoft/simm-lib
# These parameters are for comparison/validation purposes

# Regular/Low/High Vol Currency Buckets
reg_vol_ccy_bucket = ['USD', 'EUR', 'GBP', 'CHF', 'AUD', 'NZD', 'CAD', 'SEK', 'NOK', 'DKK', 'HKD', 'KRW', 'SGD', 'TWD']
low_vol_ccy_bucket = ['JPY']

# Risk Weights for Regular/Low/High Vol Currency Bucket (10-day)
# Source: InterestRateRiskWeight.java
reg_vol_rw = {
    '2w'  : 115,
    '1m'  : 112,
    '3m'  : 96,
    '6m'  : 74,
    '1y'  : 66,
    '2y'  : 61,
    '3y'  : 56,
    '5y'  : 52,
    '10y' : 53,
    '15y' : 57,
    '20y' : 60,
    '30y' : 66,
}

low_vol_rw = {
    '2w'  : 15,
    '1m'  : 18,
    '3m'  : 9,
    '6m'  : 11,
    '1y'  : 13,
    '2y'  : 15,
    '3y'  : 18,
    '5y'  : 20,
    '10y' : 19,
    '15y' : 19,
    '20y' : 20,
    '30y' : 23,
}

high_vol_rw = {
    '2w'  : 119,
    '1m'  : 93,
    '3m'  : 80,
    '6m'  : 82,
    '1y'  : 90,
    '2y'  : 92,
    '3y'  : 95,
    '5y'  : 95,
    '10y' : 94,
    '15y' : 108,
    '20y' : 105,
    '30y' : 101,
}

# Risk Weights for Inflation Rate/Cross-Currency Basis Swap Spread
inflation_rw = 63
ccy_basis_swap_spread_rw = 21

# Historical Volatility Ratio for Interest Rate Risk Class
ir_hvr = 0.53

# Vega Risk Weight for Interest Rate Risk Class
ir_vrw = 0.18

# IR Correlations - 12x12 tenor correlation matrix
ir_corr = list(
    zip(
        [1.00, 0.75, 0.63, 0.55, 0.44, 0.35, 0.31, 0.26, 0.21, 0.17, 0.15, 0.14],
        [0.75, 1.00, 0.79, 0.68, 0.51, 0.40, 0.33, 0.28, 0.22, 0.17, 0.15, 0.14],
        [0.63, 0.79, 1.00, 0.85, 0.67, 0.53, 0.45, 0.38, 0.31, 0.24, 0.22, 0.21],
        [0.55, 0.68, 0.85, 1.00, 0.83, 0.71, 0.62, 0.54, 0.45, 0.36, 0.35, 0.33],
        [0.44, 0.51, 0.67, 0.83, 1.00, 0.94, 0.86, 0.78, 0.65, 0.58, 0.55, 0.53],
        [0.35, 0.40, 0.53, 0.71, 0.94, 1.00, 0.95, 0.89, 0.78, 0.72, 0.68, 0.67],
        [0.31, 0.33, 0.45, 0.62, 0.86, 0.95, 1.00, 0.96, 0.87, 0.80, 0.76, 0.74],
        [0.26, 0.28, 0.38, 0.54, 0.78, 0.89, 0.96, 1.00, 0.94, 0.89, 0.85, 0.83],
        [0.21, 0.22, 0.31, 0.45, 0.65, 0.78, 0.87, 0.94, 1.00, 0.97, 0.94, 0.93],
        [0.17, 0.17, 0.24, 0.36, 0.58, 0.72, 0.80, 0.89, 0.97, 1.00, 0.98, 0.97],
        [0.15, 0.15, 0.22, 0.35, 0.55, 0.68, 0.76, 0.85, 0.94, 0.98, 1.00, 0.99],
        [0.14, 0.14, 0.21, 0.33, 0.53, 0.67, 0.74, 0.83, 0.93, 0.97, 0.99, 1.00],
    )
)

# Sub-curve correlation (same currency, different curves)
sub_curves_corr = 0.99

# Inflation correlation
inflation_corr = 0.37

# Cross-currency basis spread correlation
ccy_basis_spread_corr = 0.01

# Parameter for aggregating across different currencies
ir_gamma_diff_ccy = 0.24

# Credit Qualifying Risk Weights
creditQ_rw = {
    1:  75,
    2:  91,
    3:  78,
    4:  55,
    5:  67,
    6:  47,
    7:  187,
    8:  665,
    9:  262,
    10: 251,
    11: 172,
    12: 247,
    0:  665,  # Residual
}

# Vega Risk Weight for Credit Qualifying
creditQ_vrw = 0.74

# Base Correlation Weight
base_corr_weight = 8

# Credit Qualifying Correlations
creditQ_corr = [0.92, 0.41, 0.49, 0.24]

# Correlations for Credit Qualifying across different non-residual buckets
creditQ_corr_non_res = list(
    zip(
        [1.00, 0.42, 0.39, 0.39, 0.40, 0.38, 0.39, 0.34, 0.37, 0.37, 0.36, 0.31],
        [0.42, 1.00, 0.44, 0.45, 0.47, 0.45, 0.33, 0.40, 0.41, 0.40, 0.40, 0.35],
        [0.39, 0.44, 1.00, 0.43, 0.45, 0.43, 0.32, 0.35, 0.41, 0.42, 0.40, 0.35],
        [0.39, 0.45, 0.43, 1.00, 0.47, 0.44, 0.30, 0.34, 0.39, 0.43, 0.39, 0.36],
        [0.40, 0.47, 0.45, 0.47, 1.00, 0.47, 0.32, 0.35, 0.40, 0.42, 0.42, 0.37],
        [0.38, 0.45, 0.43, 0.44, 0.47, 1.00, 0.30, 0.34, 0.38, 0.40, 0.39, 0.38],
        [0.39, 0.33, 0.32, 0.30, 0.32, 0.30, 1.00, 0.28, 0.31, 0.31, 0.29, 0.26],
        [0.34, 0.40, 0.35, 0.34, 0.35, 0.34, 0.28, 1.00, 0.34, 0.35, 0.33, 0.30],
        [0.37, 0.41, 0.41, 0.39, 0.40, 0.38, 0.31, 0.34, 1.00, 0.40, 0.37, 0.32],
        [0.37, 0.40, 0.42, 0.43, 0.42, 0.40, 0.31, 0.35, 0.40, 1.00, 0.40, 0.35],
        [0.36, 0.40, 0.40, 0.39, 0.42, 0.39, 0.29, 0.33, 0.37, 0.40, 1.00, 0.34],
        [0.31, 0.35, 0.35, 0.36, 0.37, 0.38, 0.26, 0.30, 0.32, 0.35, 0.34, 1.00],
    )
)

# Credit Non-Qualifying Risk Weights
creiditNonQ_rw = {
    1: 280,
    2: 1300,
    0: 1300,  # Residual
}

# Vega Risk Weight for Credit Non-Qualifying
creditNonQ_vrw = 0.74

# Credit Non-Qualifying Correlations
creditNonQ_corr = [0.82, 0.27, 0.42]

# Correlation between non-residual buckets
cr_gamma_diff_ccy = 0.4

# Equity Risk Weights
equity_rw = {
    1:  26,
    2:  28,
    3:  34,
    4:  28,
    5:  23,
    6:  25,
    7:  29,
    8:  27,
    9:  32,
    10: 32,
    11: 18,
    12: 18,
    0:  34,  # Residual
}

# Historical Volatility Ratio for Equity
equity_hvr = 0.65

# Vega Risk Weight for Equity
equity_vrw = 0.21
equity_vrw_bucket_12 = 0.37

# Equity Correlations
equity_corr = {
    1:  0.14,
    2:  0.21,
    3:  0.27,
    4:  0.21,
    5:  0.24,
    6:  0.35,
    7:  0.34,
    8:  0.34,
    9:  0.20,
    10: 0.19,
    11: 0.45,
    12: 0.45,
    0:  0,  # Residual
}

# Equity correlations across different non-residual buckets
equity_corr_non_res = list(
    zip(
        [1.00, 0.17, 0.18, 0.17, 0.12, 0.13, 0.15, 0.13, 0.14, 0.16, 0.19, 0.19],
        [0.17, 1.00, 0.23, 0.21, 0.13, 0.15, 0.18, 0.15, 0.16, 0.18, 0.20, 0.20],
        [0.18, 0.23, 1.00, 0.24, 0.13, 0.14, 0.17, 0.15, 0.18, 0.18, 0.20, 0.20],
        [0.17, 0.21, 0.24, 1.00, 0.16, 0.18, 0.22, 0.18, 0.19, 0.21, 0.24, 0.24],
        [0.12, 0.13, 0.13, 0.16, 1.00, 0.28, 0.27, 0.26, 0.14, 0.17, 0.32, 0.32],
        [0.13, 0.15, 0.14, 0.18, 0.28, 1.00, 0.34, 0.33, 0.16, 0.19, 0.38, 0.38],
        [0.15, 0.18, 0.17, 0.22, 0.27, 0.34, 1.00, 0.32, 0.17, 0.20, 0.37, 0.37],
        [0.13, 0.15, 0.15, 0.18, 0.26, 0.33, 0.32, 1.00, 0.16, 0.19, 0.36, 0.36],
        [0.14, 0.16, 0.18, 0.19, 0.14, 0.16, 0.17, 0.16, 1.00, 0.18, 0.21, 0.21],
        [0.16, 0.18, 0.18, 0.21, 0.17, 0.19, 0.20, 0.19, 0.18, 1.00, 0.21, 0.21],
        [0.19, 0.20, 0.20, 0.24, 0.32, 0.38, 0.37, 0.36, 0.21, 0.21, 1.00, 0.45],
        [0.19, 0.20, 0.20, 0.24, 0.32, 0.38, 0.37, 0.36, 0.21, 0.21, 0.45, 1.00],
    )
)

# Commodity Risk Weights
commodity_rw = {
    1:  27,
    2:  29,
    3:  33,
    4:  25,
    5:  35,
    6:  24,
    7:  40,
    8:  53,
    9:  44,
    10: 58,
    11: 20,
    12: 21,
    13: 13,
    14: 15,
    15: 13,
    16: 58,
    17: 17,
}

# Historical Volatility Ratio for Commodity
commodity_hvr = 0.74

# Vega Risk Weight for Commodity
commodity_vrw = 0.41

# Commodity Correlations
commodity_corr = {
    1:  0.30,
    2:  0.97,
    3:  0.93,
    4:  0.98,
    5:  0.99,
    6:  0.92,
    7:  1.00,
    8:  0.58,
    9:  0.75,
    10: 0.10,
    11: 0.55,
    12: 0.64,
    13: 0.71,
    14: 0.22,
    15: 0.29,
    16: 0.00,
    17: 0.21,
}

# Commodity correlations across different non-residual buckets
commodity_corr_non_res = list(
    zip(
        [1.00, 0.18, 0.15, 0.20, 0.25, 0.08, 0.19, 0.01, 0.27, 0.00, 0.15, 0.02, 0.06, 0.07, -0.04, 0.00, 0.06],
        [0.18, 1.00, 0.89, 0.94, 0.93, 0.32, 0.22, 0.27, 0.24, 0.09, 0.45, 0.21, 0.32, 0.28, 0.17, 0.00, 0.37],
        [0.15, 0.89, 1.00, 0.87, 0.88, 0.25, 0.16, 0.19, 0.12, 0.10, 0.26, 0.07, 0.22, 0.18, 0.13, 0.00, 0.28],
        [0.20, 0.94, 0.87, 1.00, 0.92, 0.29, 0.22, 0.26, 0.24, 0.06, 0.32, 0.15, 0.26, 0.18, 0.13, 0.00, 0.34],
        [0.25, 0.93, 0.88, 0.92, 1.00, 0.30, 0.26, 0.26, 0.29, 0.10, 0.40, 0.17, 0.28, 0.24, 0.13, 0.00, 0.38],
        [0.08, 0.32, 0.25, 0.29, 0.30, 1.00, 0.35, 0.19, 0.33, 0.02, 0.23, 0.11, 0.13, 0.04, 0.06, 0.00, 0.14],
        [0.19, 0.22, 0.16, 0.22, 0.26, 0.35, 1.00, 0.11, 0.37, 0.04, 0.19, 0.05, 0.10, 0.03, -0.01, 0.00, 0.09],
        [0.01, 0.27, 0.19, 0.26, 0.26, 0.19, 0.11, 1.00, 0.18, 0.02, 0.17, 0.04, 0.07, 0.01, 0.01, 0.00, 0.07],
        [0.27, 0.24, 0.12, 0.24, 0.29, 0.33, 0.37, 0.18, 1.00, 0.06, 0.21, 0.04, 0.10, 0.06, 0.03, 0.00, 0.10],
        [0.00, 0.09, 0.10, 0.06, 0.10, 0.02, 0.04, 0.02, 0.06, 1.00, 0.12, 0.06, 0.10, 0.06, 0.07, 0.00, 0.08],
        [0.15, 0.45, 0.26, 0.32, 0.40, 0.23, 0.19, 0.17, 0.21, 0.12, 1.00, 0.34, 0.20, 0.21, 0.15, 0.00, 0.24],
        [0.02, 0.21, 0.07, 0.15, 0.17, 0.11, 0.05, 0.04, 0.04, 0.06, 0.34, 1.00, 0.19, 0.14, 0.08, 0.00, 0.15],
        [0.06, 0.32, 0.22, 0.26, 0.28, 0.13, 0.10, 0.07, 0.10, 0.10, 0.20, 0.19, 1.00, 0.34, 0.17, 0.00, 0.26],
        [0.07, 0.28, 0.18, 0.18, 0.24, 0.04, 0.03, 0.01, 0.06, 0.06, 0.21, 0.14, 0.34, 1.00, 0.26, 0.00, 0.27],
        [-0.04, 0.17, 0.13, 0.13, 0.13, 0.06, -0.01, 0.01, 0.03, 0.07, 0.15, 0.08, 0.17, 0.26, 1.00, 0.00, 0.16],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.06, 0.37, 0.28, 0.34, 0.38, 0.14, 0.09, 0.07, 0.10, 0.08, 0.24, 0.15, 0.26, 0.27, 0.16, 0.00, 1.00],
    )
)

# High FX volatility currencies
high_vol_currency_group = ['BRL', 'RUB', 'TRY', 'ZAR']

# FX Risk Weights (v2.5: 7.4, 13.6, 14.6)
fx_rw = {
    "Regular": {
        "Regular": 7.4,
        "High":    13.6,
    },
    "High": {
        "Regular": 13.6,
        "High":    14.6,
    }
}

# Historical Volatility Ratio for FX
fx_hvr = 0.6

# Vega Risk Weight for FX
fx_vrw = 0.47

# FX Correlations (Regular Vol calculation currency)
fx_reg_vol_corr = {
    "Regular": {
        "Regular": 0.5,
        "High":    0.27,
    },
    "High": {
        "Regular": 0.27,
        "High":    0.42,
    }
}

# FX Correlations (High Vol calculation currency)
fx_high_vol_corr = {
    "Regular": {
        "Regular": 0.85,
        "High":    0.69,
    },
    "High": {
        "Regular": 0.69,
        "High":    0.5,
    }
}

# FX Vega Correlation
fx_vega_corr = 0.5

# Delta Concentration Thresholds for Interest Rate Risk
ir_delta_CT = {
    'Others' : 33,   # High volatility currencies
    'USD'    : 230,  # Regular volatility, well-traded
    'EUR'    : 230,  # Regular volatility, well-traded
    'GBP'    : 230,  # Regular volatility, well-traded
    'AUD'    : 44,   # Regular volatility, less well-traded
    'CAD'    : 44,   # Regular volatility, less well-traded
    'CHF'    : 44,   # Regular volatility, less well-traded
    'DKK'    : 44,   # Regular volatility, less well-traded
    'HKD'    : 44,   # Regular volatility, less well-traded
    'KRW'    : 44,   # Regular volatility, less well-traded
    'NOK'    : 44,   # Regular volatility, less well-traded
    'NZD'    : 44,   # Regular volatility, less well-traded
    'SEK'    : 44,   # Regular volatility, less well-traded
    'SGD'    : 44,   # Regular volatility, less well-traded
    'TWD'    : 44,   # Regular volatility, less well-traded
    'JPY'    : 70,   # Low volatility
}

# Delta Concentration Thresholds for Credit Spread Risk
credit_delta_CT = {
    "Qualifying" : {
        1 : 1.00,
        2 : 0.25,
        3 : 0.25,
        4 : 0.25,
        5 : 0.25,
        6 : 0.25,
        7 : 1.00,
        8 : 0.25,
        9 : 0.25,
        10: 0.25,
        11: 0.25,
        12: 0.25,
        0 : 0.25,  # Residual
    },
    "Non-Qualifying" : {
        1 : 9.5,
        2 : 0.5,
        0 : 0.5,  # Residual
    }
}

# Delta Concentration Thresholds for Equity Risk
equity_delta_CT = {
    1 : 8,
    2 : 8,
    3 : 8,
    4 : 8,
    5 : 22,
    6 : 22,
    7 : 22,
    8 : 22,
    9 : 0.27,
    10: 0.60,
    11: 1200,
    12: 1200,
    0 : 0.27,  # Residual
}

# Delta Concentration Thresholds for Commodity Risk
commodity_delta_CT = {
    1 : 140,
    2 : 2000,
    3 : 310,
    4 : 310,
    5 : 310,
    6 : 2600,
    7 : 2600,
    8 : 750,
    9 : 750,
    10: 52,
    11: 490,
    12: 1300,
    13: 72,
    14: 72,
    15: 72,
    16: 52,
    17: 4200,
}

# Currency Categories for FX
fx_category1 = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CHF', 'CAD']
fx_category2 = ['BRL', 'CNY', 'HKD', 'INR', 'KRW', 'MXN', 'NOK', 'NZD', 'RUB', 'SEK', 'SGD', 'TRY', 'ZAR']

# Delta Concentration Thresholds for FX Risk
fx_delta_CT = {
    'Category1' : 8400,
    'Category2' : 1900,
    'Others'    : 560,
}

# Vega Concentration Thresholds for IR Risk
ir_vega_CT = {
    'Others' : 120,  # High volatility
    'USD'    : 3300, # Regular volatility, well-traded
    'EUR'    : 3300, # Regular volatility, well-traded
    'GBP'    : 3300, # Regular volatility, well-traded
    'AUD'    : 470,  # Regular volatility, less well-traded
    'CAD'    : 470,  # Regular volatility, less well-traded
    'CHF'    : 470,  # Regular volatility, less well-traded
    'DKK'    : 470,  # Regular volatility, less well-traded
    'HKD'    : 470,  # Regular volatility, less well-traded
    'KRW'    : 470,  # Regular volatility, less well-traded
    'NOK'    : 470,  # Regular volatility, less well-traded
    'NZD'    : 470,  # Regular volatility, less well-traded
    'SEK'    : 470,  # Regular volatility, less well-traded
    'SGD'    : 470,  # Regular volatility, less well-traded
    'TWD'    : 470,  # Regular volatility, less well-traded
    'JPY'    : 570,  # Low volatility
}

# Vega Concentration Thresholds for Credit Spread Risk
credit_vega_CT = {
    "Qualifying"     : 310,
    "Non-Qualifying" : 150,
}

# Vega Concentration Thresholds for Equity Risk
equity_vega_CT = {
    1 : 130,
    2 : 130,
    3 : 130,
    4 : 130,
    5 : 7400,
    6 : 7400,
    7 : 7400,
    8 : 7400,
    9 : 300,
    10: 190,
    11: 9400,
    12: 9400,
    0 : 300,  # Residual
}

# Vega Concentration Thresholds for Commodity Risk
commodity_vega_CT = {
    1 : 250,
    2 : 2000,
    3 : 510,
    4 : 510,
    5 : 510,
    6 : 2700,
    7 : 2700,
    8 : 870,
    9 : 870,
    10: 220,
    11: 450,
    12: 740,
    13: 380,
    14: 380,
    15: 380,
    16: 480,
    17: 79,
}

# Vega Concentration Thresholds for FX Risk
fx_vega_CT = {
    'Category1-Category1' : 2700,
    'Category1-Category2' : 1200,
    'Category1-Category3' : 610,
    'Category2-Category2' : 570,
    'Category2-Category3' : 370,
    'Category3-Category3' : 170,
}

# Correlation between Risk Classes within Product Classes
corr_params = list(
    zip(
        [1.00, 0.28, 0.18, 0.18, 0.30, 0.22],
        [0.28, 1.00, 0.30, 0.66, 0.46, 0.27],
        [0.18, 0.30, 1.00, 0.46, 0.32, 0.18],
        [0.18, 0.66, 0.46, 1.00, 0.63, 0.24],
        [0.30, 0.46, 0.32, 0.63, 1.00, 0.26],
        [0.22, 0.27, 0.18, 0.24, 0.26, 1.00],
    )
)
