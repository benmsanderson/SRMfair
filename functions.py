from fair.tools.ensemble import tcrecs_generate
from fair.forward import fair_scm
import fair
from scipy import stats
import numpy as np
import sys
import pandas as pd

def print_progress_bar(index, total, label):
    """
    Print a progress bar to the console.

    Parameters:
    ----------
    index : int
        Current progress index.
    total : int
        Total number of steps.
    label : str
        Label to display next to the progress bar.
    """
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()


def make_params(n=100):
    """
    Generate parameter sets for the FaIR model.

    Parameters:
    ----------
    n : int
        Number of parameter sets to generate.

    Returns:
    -------
    pmat : pd.DataFrame
        DataFrame containing the generated parameter sets.
    """
    samples = n

    # generate some joint lognormal TCR and ECS pairs
    tcrecs = tcrecs_generate(n=samples, seed=38571)

    # generate some forcing scale factors with SD of 10% of the best estimate
    # Chris: this is over-constrained and probably just something from my example
    # Instead let's repeat what we did in FaIR 1.3 code, component by component
    # using AR5 scalings
    zscore = stats.norm.ppf(0.95)
    scales1d = np.array(
        [
            0.2,      # CO2
            0.28,     # CH4: updated value from etminan 2016
            0.2,      # N2O
            0.2,      # other WMGHS
            0.4-0.2,        # tropospheric O3
            -0.05-(-0.15),  # stratospheric O3
            0.07-0.02,      # stratospheric WV from CH4
            1,              # contrails (lognormal)
            0.8/0.9,  # aerosols
            1,      # black carbon on snow (lognormal)
            -0.15-(-0.25),  # land use change
            1.0-0.5,        # volcanic
            0.05
        ]
    )/zscore # solar (additive)
    scales2d = np.repeat(scales1d[np.newaxis,:],samples,axis=0)

    locs = np.array([1,1,1,1,0.4,-0.05,0.07,1,1,1,-0.15,1.0,0.00])
    locs2d = np.repeat(locs[np.newaxis,:],samples,axis=0)

    # BC-snow and contrails are lognormal with sigma=0.5 and sigma=0.65: see page 8SM-11
    F_scale = stats.norm.rvs(size=(samples,13), loc=locs2d[:,:13], scale=scales2d[:,:13], random_state=40000)
    F_scale[:,9] = stats.lognorm.rvs(0.5, size=samples, random_state=40001)
    F_scale[:,7]  = stats.lognorm.rvs(0.65, size=samples, random_state=40002)

    # aerosols are asymmetric Gaussian
    F_scale[F_scale[:,8]<-0.9,8] = 1./0.8*(F_scale[F_scale[:,8]<-0.9,8]+0.9) - 0.9


    #F_scale = stats.norm.rvs(size=(samples,13), loc=1, scale=0.1, random_state=40000)

    # do the same for the carbon cycle parameters
    r0 = stats.norm.rvs(size=samples, loc=35, scale=3.5, random_state=41000)
    rc = stats.norm.rvs(size=samples, loc=0.019, scale=0.0019, random_state=42000)
    rt = stats.norm.rvs(size=samples, loc=4.165, scale=0.4165, random_state=45000)


    pmatdict=[]
    for i in range(n):
        p={}
        p['tcrecs']=tcrecs[i]
        p['F_scale']=F_scale[i,:]
        p['r0']=r0[i]
        p['rc']=rc[i]
        p['rt']=rt[i]
        pmatdict.append(p)
    pmat=pd.DataFrame(pmatdict)
    return pmat

def run_fair(emissions, pmat0=None, useMultigas=True,other_rf=None):
    """
    Run the FaIR model with the given emissions and parameters.

    Parameters:
    ----------
    emissions : array-like
        Emissions data.
    pmat0 : pd.DataFrame, optional
        DataFrame containing the parameter sets.
    useMultigas : bool, optional
        Whether to use the multigas configuration.
    other_rf : array-like, optional
        Other radiative forcing data.

    Returns:
    -------
    Ce : np.ndarray
        Concentrations.
    Fe : np.ndarray
        Forcings.
    Te : np.ndarray
        Temperatures.
    """
    if type(pmat0)==type(None):
        pl=[]
        pdc={}
        pdc['tcrecs']=np.array([1.6,2.75])
        pdc['F_scale']=np.ones(13)
        pdc['r0']=35
        pdc['rc']=0.02
        pdc['rt']=4.1
        pdc['F2x']=3.73
        pl.append(pdc)
        pmat0=pd.DataFrame(pl)
    samples = len(pmat0)
   
    Te=np.zeros((len(emissions),samples))

    if useMultigas:
        Ce=np.zeros((len(emissions),31,samples))
        Fe=np.zeros((len(emissions),13,samples))
        for i in range(samples):
        
            Ce[:,:,i], Fe[:,:,i], Te[:,i] = fair.forward.fair_scm(emissions=emissions,
                                r0 = pmat0.iloc[i]['r0'],
                                rc = pmat0.iloc[i]['rc'],
                                rt = pmat0.iloc[i]['rt'],
                                tcrecs = np.array(pmat0.iloc[i]['tcrecs'][:]),
                                scale = np.array(pmat0.iloc[i]['F_scale'][:]))
    else:
        Ce=np.zeros((len(emissions),samples))
        Fe=np.zeros((len(emissions),samples))
        for i in range(samples):
            Ce[:,i], Fe[:,i], Te[:,i] = fair.forward.fair_scm(emissions=emissions,other_rf=other_rf,useMultigas=False,
                            r0 = pmat0.iloc[i]['r0'],
                            rc = pmat0.iloc[i]['rc'],
                            rt = pmat0.iloc[i]['rt'],
                            tcrecs = np.array(pmat0.iloc[i]['tcrecs'][:]),
                            scale = pmat0.iloc[i]['F_scale'][0])
    Ce=np.squeeze(Ce)
    Fe=np.squeeze(Fe)
    Te=np.squeeze(Te)
    return Ce,Fe,Te


def constrain_params(pmat):
    """
    Constrain the parameter sets based on historical temperature data.

    Parameters:
    ----------
    pmat : pd.DataFrame
        DataFrame containing the parameter sets.

    Returns:
    -------
    pmat : pd.DataFrame
        Constrained parameter sets.
    """
    #Follow Smith et al to constrain output based on CW
    try:
        # For Python 3.0 and later
        from urllib.request import urlopen
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen

    from fair.tools.constrain import hist_temp

    # load up Cowtan and Way data remotely
    url = 'http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_annual_v2_0_0.txt'
    response = urlopen(url)

    CW = np.loadtxt(response)
    constrained = np.zeros(samples, dtype=bool)

    for i in range(samples):
        # we use observed trend from 1880 to 2016
        constrained[i], _, _, _, _ = hist_temp(CW[30:167,1], Te[1880-1765:2017-1765,i], CW[30:167,0])

    # How many ensemble members passed the constraint?
    print('%d ensemble members passed historical constraint' % np.sum(constrained))
    pmat=[pmat0[i] for i in range(samples) if constrained[i]]


def generate_efficacy(mean, length):
    """
    Generate a random sequence of SRM efficacy values.

    Parameters:
    ----------
    mean : float
        Mean efficacy value.
    length : int
        Length of the sequence.

    Returns:
    -------
    sequence : np.ndarray
        Generated sequence of efficacy values.
    """
        # Ensure the mean is between 0 and 1
    if not (0 <= mean <= 1):
        raise ValueError("Mean must be between 0 and 1")
    
    # Generate a random sequence
    alpha = mean * 10
    beta = (1 - mean) * 10
    sequence = np.random.beta(alpha, beta, length)
    
    return sequence


def find_consecutive_zeros(arr):
    """
    Find consecutive zeros in an array.

    Parameters:
    ----------
    arr : array-like
        Input array.

    Returns:
    -------
    consecutive_zeros : list of lists
        List of indices of consecutive zeros.
    """
    consecutive_zeros = []
    current_zeros = []

    for i, value in enumerate(arr):
        if value == 0:
            current_zeros.append(i)
        else:
            if current_zeros:
                consecutive_zeros.append(current_zeros)
                current_zeros = []

    if current_zeros:
        consecutive_zeros.append(current_zeros)

    return consecutive_zeros

import numpy as np

def simulate_failure(prob_failure=0.01, avg_outage_length=10, total_years=100,mean_efficacy=0.001):
    """
    Simulate SRM failure events and their impact on efficacy.

    Parameters:
    ----------
    prob_failure : float
        Annual probability of SRM failure.
    avg_outage_length : int
        Average length of SRM outages (years).
    total_years : int
        Total number of years to simulate.
    mean_efficacy : float
        Mean efficacy of SRM during non-failure periods.

    Returns:
    -------
    failure_sequence : np.ndarray
        Sequence of SRM failure events (1 for failure, 0 for no failure).
    fracfail : np.ndarray
        Fractional efficacy during non-failure periods.
    """
    if not (0 <= mean_efficacy <= 1):
        raise ValueError("Mean must be between 0 and 1")
    alpha = mean_efficacy * 10
    beta = (1 - mean_efficacy) * 10

    failures = np.random.rand(total_years) < prob_failure
    outage_lengths = np.random.poisson(avg_outage_length, total_years)
    efficacy = np.random.beta(alpha, beta, total_years)
    failure_sequence = np.zeros(total_years)
    fracfail = np.zeros(total_years)+1
    
    for i in range(total_years):
        if failures[i]:
            failure_sequence[i:i+outage_lengths[i]] = 1
    list_periods=find_consecutive_zeros(failure_sequence)
    for i in range(len(list_periods)):
        if len(list_periods[i])>0:
            fracfail[list_periods[i]]=efficacy[list_periods[i][0]]
    
    return failure_sequence,fracfail




def adpt_fair(ems,sint,threshold,df,wd=2500,wf=1,i=0,p=None,iters=100):
    """
    Adaptively deploy SRM to maintain temperature below a threshold.

    Parameters:
    ----------
    ems : object
        Emissions data.
    sint : float
        Sensitivity parameter for SRM adjustment.
    threshold : float
        Temperature threshold to maintain.
    df : pd.DataFrame
        DataFrame containing SRM configuration parameters.
    wd : int, optional
        Width parameter for SRM adjustment.
    wf : int, optional
        Weight factor for SRM adjustment.
    i : int, optional
        Index of the SRM configuration to use.
    p : pd.DataFrame, optional
        DataFrame containing the parameter sets.
    iters : int, optional
        Number of iterations for SRM adjustment.

    Returns:
    -------
    Ctmp1 : np.ndarray
        Concentrations.
    Ftmp1 : np.ndarray
        Forcings.
    Ttmp1 : np.ndarray
        Temperatures.
    srm_act : np.ndarray
        SRM activity.
    ems1 : np.ndarray
        Adjusted emissions.
    T45g0 : np.ndarray
        Baseline temperatures.
    """
    Ce, Fe, Te = run_fair(ems.Emissions.emissions,pmat0=p)
    ems_bs=np.sum(ems.Emissions.emissions[:,1:3],axis=1)
    f_bs=np.sum(Fe[:,1:11],axis=1)

    C45g0, F45g0, T45g0 = run_fair(ems_bs,pmat0=p,other_rf=f_bs,useMultigas=False)


    Ttmp1=T45g0
    srm1=f_bs*0
    de=srm1
    srm_on=srm1
    srm_act=srm1
    
    
    istart=int(df.loc[i]['Start']-ems.Emissions.emissions[0,0])
    iend=int(df.loc[i]['End']-ems.Emissions.emissions[0,0])
    nyrs=iend-istart
    failure_sequence,fracfail = simulate_failure(df.loc[i]['pfail'], df.loc[i]['aol'], nyrs, df.loc[i]['Effic'])
    srm_on[istart:iend]=1-failure_sequence[:]
    ifade=int(df.loc[i]['fade'])

    ems1=ems_bs.copy()
    for j in np.arange(0,iters):

        Ctmp1, Ftmp1, Ttmp1 = run_fair(ems1,pmat0=p,other_rf=f_bs+srm1,useMultigas=False)

        ovsht=(Ttmp1-threshold).clip(min=0)
        srm1=((srm1-ovsht/sint)*srm_on).clip(min=-df.loc[i]['maxsrm']).clip(max=0)
        srm_g0=srm1*0
        srm_g0[srm1<-0.1]=1
        if j<10:
        ems1=np.interp(np.arange(0,len(srm1),1)-np.cumsum(srm_g0*srm_on*(1-np.sign(np.diff(ems1, prepend=ems1[0])))/2)*df.mhaz[i],np.arange(0,len(srm1),1),ems_bs)
    srm_act[istart:iend]=srm1[istart:iend]*(fracfail)
    if ifade>0:
        srm_act[iend:(iend+ifade)]=srm_act[iend]*(1-np.arange(0,ifade)/ifade)
    

    Ctmp1, Ftmp1, Ttmp1 = run_fair(ems1,pmat0=p,other_rf=f_bs+srm_act,useMultigas=False)

    return Ctmp1, Ftmp1, Ttmp1, srm_act, ems1, T45g0

def compute_damages(temperature_series, dt=1.0, a=0.002, b=0.001, c=0.0005, D0=0.0):
    """
    Compute time series of damages as a function of global mean temperature anomaly and its rate of change.
    
    Parameters:
    ----------
    temperature_series : array-like or pd.Series
        Time series of global mean temperature anomalies (°C).
    dt : float
        Time step between temperature observations (years). Default is 1.0 (annual data).
    a, b, c : float
        Coefficients for temperature level, rate of change, and interaction terms.
    D0 : float
        Baseline damage (can be set to 0 if not needed).
        
    Returns:
    -------
    damages : np.ndarray
        Time series of computed damages.
    dT_dt : np.ndarray
        Rate of change of temperature anomaly.
    """
    
    T = np.asarray(temperature_series)
    
    # Compute rate of temperature change (central differences)
    dT_dt = np.zeros_like(T)
    dT_dt[1:-1] = np.abs(T[2:] - T[:-2]) / (2 * dt)
    dT_dt[0] = (T[1] - T[0]) / dt  # Forward difference at start
    dT_dt[-1] = (T[-1] - T[-2]) / dt  # Backward difference at end
    
    # Compute damages
    damages = D0 + a * T**2 + b * dT_dt**2 + c * T * dT_dt
    
    return damages, dT_dt

def calc_damages(T,damage_parameter_sets):
    """
    Calculate damages for multiple parameter sets.

    Parameters:
    ----------
    T : array-like
        Temperature time series.
    damage_parameter_sets : dict
        Dictionary of damage parameter sets.

    Returns:
    -------
    damages : dict
        Dictionary of calculated damages for each parameter set.
    """
    damages = {}
    for key, params in damage_parameter_sets.items():
        damages[key], dT_dt = compute_damages(T, a=params["a"], b=params["b"], c=params["c"], D0=params["D0"])

    return damages

def integrate_damages(
    damages,
    method="standard",
    dt=1.0,
    rho_const=0.02,              # Standard IAM constant discount rate
    rho_0_ramsey=0.001,          # Ramsey pure rate of time preference
    eta_ramsey=1.,              # Ramsey elasticity of marginal utility
    growth_rate=0.007,            # Per capita consumption growth rate
    tstart=260
):
    """
    Integrate climate damages over time with two alternative discounting approaches.
    
    Parameters:
    ----------
    damages : array-like
        Time series of damages as fraction of GDP.
    method : str
        Discounting method: "standard" (constant rate) or "ethical" (Ramsey-type).
    dt : float
        Time step (years).
    rho_const : float
        Constant discount rate (used if method="standard").
    rho_0_ramsey : float
        Pure time preference rate (used if method="ethical").
    eta_ramsey : float
        Elasticity of marginal utility of consumption (used if method="ethical").
    growth_rate : float
        Per capita consumption growth rate (used if method="ethical").
    
    Returns:
    -------
    pv_damages : float
        Present value of integrated damages (as fraction of GDP).
    discount_factors : np.ndarray
        Discount factor time series.
    """
    damages = np.asarray(damages)
    #T = np.tile(np.arange(len(damages)) * dt, (damages.shape[1],1))
    T = np.arange(len(damages)) * dt
    if method == "standard":
        # Constant discounting
        discount_factors = np.exp(-rho_const * T)
    
    elif method == "ethical":
        # Ramsey discounting
        rho_t = rho_0_ramsey + eta_ramsey * growth_rate
        cumulative_discount = np.cumsum(np.full_like(T, rho_t) * dt)
        discount_factors = np.exp(-cumulative_discount)
    
    else:
        raise ValueError("Invalid method. Choose 'standard' or 'ethical'.")
    if damages.ndim>1:
        discount_factors=np.repeat(discount_factors[:,None], damages.shape[1], axis=1)
    # Present value of damages
    pv_damages = np.sum(damages[tstart:] * discount_factors[:len(damages[tstart:])],axis=0) * dt
    
    return pv_damages, discount_factors

def icalc_damages(T,damage_parameter_sets):
    idamages = {}
    damages = calc_damages(T,damage_parameter_sets)
    for key in damage_parameter_sets.keys():
        idamages[key+'_standard'], discount_factors = integrate_damages(damages[key], method="standard")
    for key in damage_parameter_sets.keys():

        idamages[key+'_ethical'], discount_factors = integrate_damages(damages[key], method="ethical")

    return idamages