import numpy as np

# Constants from kinds.f90 / master_parameters.f90 / computeFlowRule_otis.f90
WP = np.float64
ZERO = 0.0
ONE = 1.0
TWO = 2.0
SMALL_PARAM = 1.0e-20

class AutoDiff:
    """
    Python implementation of ONUMM6N1 for forward-mode automatic differentiation.
    Tracks value (val) and an array of derivatives (grad).
    """
    __slots__ = ['val', 'grad']

    def __init__(self, val, grad=None, size=6):
        self.val = float(val)
        if grad is None:
            self.grad = np.zeros(size, dtype=WP)
        else:
            self.grad = np.array(grad, dtype=WP)

    def __repr__(self):
        return f"AutoDiff(val={self.val}, grad={self.grad})"

    def __add__(self, other):
        if isinstance(other, AutoDiff):
            return AutoDiff(self.val + other.val, self.grad + other.grad)
        else:
            return AutoDiff(self.val + other, self.grad)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, AutoDiff):
            return AutoDiff(self.val - other.val, self.grad - other.grad)
        else:
            return AutoDiff(self.val - other, self.grad)

    def __rsub__(self, other):
        if isinstance(other, AutoDiff):
            return AutoDiff(other.val - self.val, other.grad - self.grad)
        else:
            return AutoDiff(other - self.val, -self.grad)

    def __mul__(self, other):
        if isinstance(other, AutoDiff):
            # Product rule: (u*v)' = u'v + uv'
            return AutoDiff(self.val * other.val, self.grad * other.val + self.val * other.grad)
        else:
            return AutoDiff(self.val * other, self.grad * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, AutoDiff):
            # Quotient rule: (u/v)' = (u'v - uv') / v^2
            denom = other.val ** 2
            return AutoDiff(self.val / other.val, (self.grad * other.val - self.val * other.grad) / denom)
        else:
            return AutoDiff(self.val / other, self.grad / other)

    def __rtruediv__(self, other):
        # (c/v)' = -c v' / v^2
        if isinstance(other, AutoDiff):
             # Handled by __truediv__
             return other.__truediv__(self)
        else:
            denom = self.val ** 2
            return AutoDiff(other / self.val, -other * self.grad / denom)

    def __pow__(self, power):
        # Power rule could be complex if power is AutoDiff, but usually it's scalar constant in this physics
        if isinstance(power, AutoDiff):
            # u^v = exp(v * ln(u)) -> too complex if not needed.
            # Assuming power is scalar for this specific implementation based on F90 code
             raise NotImplementedError("Power with AutoDiff exponent not implemented yet")
        else:
            # (u^n)' = n * u^(n-1) * u'
            if self.val == 0 and power < 1:
                # Handle singularity if needed, for now let it error or produce Inf
                new_val = 0.0
                new_grad = np.zeros_like(self.grad)
            else:
                new_val = self.val ** power
                new_grad = power * (self.val ** (power - 1)) * self.grad
            return AutoDiff(new_val, new_grad)

    def __lt__(self, other):
        return self.val < (other.val if isinstance(other, AutoDiff) else other)
    
    def __gt__(self, other):
        return self.val > (other.val if isinstance(other, AutoDiff) else other)
        
    def __ge__(self, other):
        return self.val >= (other.val if isinstance(other, AutoDiff) else other)

def ad_exp(x):
    if isinstance(x, AutoDiff):
        # exp(u)' = exp(u) * u'
        e_val = np.exp(x.val)
        return AutoDiff(e_val, e_val * x.grad)
    return np.exp(x)

def ad_sqrt(x):
    if isinstance(x, AutoDiff):
        # sqrt(u)' = 0.5/sqrt(u) * u'
        sqrt_val = np.sqrt(x.val)
        if sqrt_val == 0:
             return AutoDiff(0.0, np.zeros_like(x.grad))
        return AutoDiff(sqrt_val, (0.5 / sqrt_val) * x.grad)
    return np.sqrt(x)

def ad_sign(x):
    # sign is constant derivative 0 almost everywhere
    if isinstance(x, AutoDiff):
        return np.sign(x.val)
    return np.sign(x)


def compute_flow_rule(
    stress_shear_resolved,  # 12 floats
    temperature_current,    # float
    coeff_degrad_plastic_energy=1.0,  # float
    # Resistances (12 floats each)
    res_backstress_undamaged=None,
    res_thermal_ssd_undamaged=None,
    res_thermal_cross_slip_undamaged=None,
    res_athermal_density_independent=None,
    res_athermal_ssd_undamaged=None,
    res_thermal_gnd_undamaged=None,
    res_athermal_gnd_undamaged=None,
    increment_time=1.0
):
    # Default values matching driver_otis.f90
    if res_backstress_undamaged is None: res_backstress_undamaged = np.zeros(12)
    if res_thermal_ssd_undamaged is None: res_thermal_ssd_undamaged = np.array([
        539422634.907665, 539422634.907665, 539422634.907665,
        512710648.819887, 512710648.819887, 512710648.819887,
        514299352.207645, 514299352.207645, 514299352.207645,
        538152337.407513, 538152337.407513, 538152337.407513 ])
    if res_thermal_cross_slip_undamaged is None: res_thermal_cross_slip_undamaged = np.full(12, 297692600.395148)
    if res_athermal_density_independent is None: res_athermal_density_independent = np.zeros(12)
    if res_athermal_ssd_undamaged is None: res_athermal_ssd_undamaged = np.zeros(12)
    if res_thermal_gnd_undamaged is None: res_thermal_gnd_undamaged = np.zeros(12)
    if res_athermal_gnd_undamaged is None: res_athermal_gnd_undamaged = np.zeros(12)

    # Constants
    rate_slip_reference = 1.0e7
    energy_activation = 9.5e-19
    constant_barrier_shape_p = 0.78
    constant_barrier_shape_q = 1.15
    res_thermal_density_independent = 316.7e6
    constant_boltzmann = 1.3806e-23
    
    n_slip_systems = 12
    
    # Results
    results = {
         'Increment_Slip_Plastic': np.zeros(12),
         'dInc_dStress': np.zeros(12),
         'dInc_dBackstress': np.zeros(12),
         'dInc_dThermalSsd': np.zeros(12),
         'dInc_dThermalCrossSlip': np.zeros(12),
         'dInc_dAthermalSsd': np.zeros(12)
    }

    for i in range(n_slip_systems):
        # E1: d/dStressResolved
        # E2: d/dBackStressUndamaged
        # E3: d/dThermalSsdUndamaged
        # E4: d/dThermalCrossSlipUndamaged
        # E5: d/dAthermalSsdUndamaged
        
        # Initialize AD variables
        stress_ot = AutoDiff(stress_shear_resolved[i])
        stress_ot.grad[0] = 1.0  # E1
        
        res_bs_ot = AutoDiff(res_backstress_undamaged[i])
        res_bs_ot.grad[1] = 1.0 # E2
        
        res_th_ssd_ot = AutoDiff(res_thermal_ssd_undamaged[i])
        res_th_ssd_ot.grad[2] = 1.0 # E3
        if res_th_ssd_ot.val < SMALL_PARAM: res_th_ssd_ot.val = SMALL_PARAM
            
        res_th_cs_ot = AutoDiff(res_thermal_cross_slip_undamaged[i])
        res_th_cs_ot.grad[3] = 1.0 # E4
        if res_th_cs_ot.val < SMALL_PARAM: res_th_cs_ot.val = SMALL_PARAM
            
        res_ath_ssd_ot = AutoDiff(res_athermal_ssd_undamaged[i])
        res_ath_ssd_ot.grad[4] = 1.0 # E5
        if res_ath_ssd_ot.val < SMALL_PARAM: res_ath_ssd_ot.val = SMALL_PARAM

        # Logic
        res_bs_new_ot = res_bs_ot * coeff_degrad_plastic_energy
        
        stress_driving_ot = stress_ot - res_bs_new_ot
        
        # Athermal
        term1 = (res_ath_ssd_ot ** TWO) + (res_athermal_gnd_undamaged[i] ** TWO)
        res_ath_total_ot = (ad_sqrt(term1) + res_athermal_density_independent[i]) * coeff_degrad_plastic_energy
        
        # Thermal
        term2 = (res_th_ssd_ot ** TWO) + (res_thermal_gnd_undamaged[i] ** TWO) + (res_th_cs_ot ** TWO)
        res_th_total_ot = (ad_sqrt(term2) + res_thermal_density_independent) * coeff_degrad_plastic_energy
        
        stress_sign = np.sign(stress_driving_ot.val)
        if stress_sign == 0: stress_sign = 1.0
        
        # Effective stress (Driving - Athermal) * sign
        stress_effective_ot = (stress_driving_ot * stress_sign) - res_ath_total_ot
        
        inc_slip = AutoDiff(0.0)
        
        if stress_effective_ot.val > ZERO:
            ratio_stress_ot = stress_effective_ot / res_th_total_ot
            
            if ratio_stress_ot.val >= ONE:
                 inc_slip = AutoDiff(rate_slip_reference) * stress_sign * increment_time
                 # Note: In Fortran code, max slip deriv is 0? The code just assigns rateSlipRef.
                 # Wait, F90 says: incrementSlipPlastic_ot = rateSlipReference * sign * incrementTime
                 # If rateSlipReference is constant, derivative is 0.
                 # No, derivatives propagate via `sign`. But sign is const derivative.
                 # So derivative is 0.
            else:
                 p = constant_barrier_shape_p
                 q = constant_barrier_shape_q
                 kT = constant_boltzmann * temperature_current
                 rate_ref = rate_slip_reference
                 
                 # (1 - ratio^p)
                 term3 = ONE - (ratio_stress_ot ** p)
                 
                 # term3^q
                 term3 = term3 ** q
                 
                 # exponent arg
                 term3 = term3 * (-energy_activation / kT)
                 
                 inc_slip = AutoDiff(rate_ref) * ad_exp(term3) * stress_sign * increment_time
        
        results['Increment_Slip_Plastic'][i] = inc_slip.val
        results['dInc_dStress'][i] = inc_slip.grad[0]
        results['dInc_dBackstress'][i] = inc_slip.grad[1]
        results['dInc_dThermalSsd'][i] = inc_slip.grad[2]
        results['dInc_dThermalCrossSlip'][i] = inc_slip.grad[3]
        results['dInc_dAthermalSsd'][i] = inc_slip.grad[4]

    return results

if __name__ == "__main__":
    # Test with default values from driver_otis.f90
    print("Running Python Physics Engine Test...")
    stress_default = np.array([351341296.634225, -18069556.4068816, -333271740.227343, 
                               353247960.639986, -16258249.2359092, -336989711.404077, 
                               348287762.056055, 3723893.08626250, -352011655.142318, 
                               353955103.276707, 1921201.35165965, -355876304.628367])
    
    res = compute_flow_rule(stress_default, 1123.0)
    import pandas as pd
    df = pd.DataFrame(res)
    print(df.head())
    df.to_csv('output_otis.csv', index=False)
