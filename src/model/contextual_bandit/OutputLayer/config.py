config = {
    'lower_thresholds': {
        'nr. sessions': 0.0,
        'total km': 2.0,
        'km Z3-4': 0.05,
        'km Z5-T1-T2': 0.025,
        'km sprinting': 0.0125,
        'strength training' : 0.5,
        'hours alternative': 0.5
    },
    'upper_thresholds': {
        'ratio_of_total_km': {
            'km Z3-4': 0.15,
            'km Z5-T1-T2': 0.15,
            'km sprinting': 0.075
        },
        'increase_vs_history': {
            'total km': 5.0,
            'hours alternative': 1.0
        },
        'absolute': {
            'nr. sessions': 1.0,
            'strength training' : 1.0
        },
        'days_for_historic_comparison': 28
    }
}