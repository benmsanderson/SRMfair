from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import os as os
def setup_fair():
    # %%
    f = FAIR()

    # %%
    snames=["high-extension", "high-overshoot", "medium-extension", "medium-overshoot", "low", "verylow", "verylow-overshoot"]
    snames_short=["H-ext","H-ext-OS","M-ext","ML-ext","L-ext","VLLO-ext","VLHO-ext"]
    sname21_short=["H","H","M","ML","L","VLLO","VLHO"]


    f.define_time(1750, 2501, 1)
    f.define_scenarios(snames)
    species, properties = read_properties('../data/fair-parameters/species_configs_properties_1.4.1.csv')
    f.define_species(species, properties)
    f.ch4_method='Thornhill2021'
    df_configs = pd.read_csv('../data/fair-parameters/calibrated_constrained_parameters_1.4.1.csv', index_col=0)
    f.define_configs(df_configs.index)
    f.allocate()

    # %%
    scens=f.emissions.scenario.values

    # %%
    ldict={}
    ldict21={}
    for i,s in enumerate(snames):
        ldict[s]=snames_short[i]
        ldict21[s]=sname21_short[i]


    # %%
    colors = {
        snames[0]: '#800000',
        snames[1]: '#ff0000',
        snames[2]: '#fc7b03',
        snames[3]: '#d3a640',
        snames[4]: '#098740',
        snames[5]: '#0080d0',
        snames[6]: '#100060',
    }

    # %%
    os.makedirs('../plots', exist_ok=True)

    # %%
    df_emis = pd.read_csv('../data/emissions/extensions_1750-2500.csv')
    gwpmat=df_emis.loc[df_emis.scenario=='verylow-overshoot'].set_index('variable')['ar6_gwp_mass_adjusted']

    # %%
    ch4dict={}
    ch4dict[scens[0]]=[[2200,2250],[np.nan,50]] # H-ext
    ch4dict[scens[1]]=[[2100,2200],[np.nan,50]] # H-ext-OS
    ch4dict[scens[2]]=[[2300],[np.nan]]#    M-ext
    ch4dict[scens[3]]=[[2040,2050,2060,2125,2150,2300],[np.nan,350,300,170,120,120]]#    MOS-ext
    #ch4dict[scens[4]]=[[2300],[np.nan]]#    L-ext
    ch4dict[scens[4]]=[[2025,2030,2040,2050,2060,2080,2090,2100,2125,2300],[np.nan,260,200,150,110,100,95,95,95,95]]#    VL-ext
    ch4dict[scens[5]]=[[2025,2030,2040,2050,2060,2080,2090,2100,2125,2300],[np.nan,260,150,120,100,95,90,90,90,90]]#    VL-ext
    ch4dict[scens[6]]=[[2025,2030,2050,2060,2100,2300],[np.nan,360,300,170,120,120]]#    L-ext-OS

    # %%
    n2odict={}
    n2odict[scens[0]]=[[2200,2250],[np.nan,6]] # H-ext
    n2odict[scens[1]]=[[2100,2200],[np.nan,6]] # H-ext-OS
    n2odict[scens[2]]=[[2300],[np.nan]] #    M-ext
    n2odict[scens[3]]=[[2300],[np.nan]]     #    MOS-ext
    n2odict[scens[4]]=[[2300],[np.nan]]   #    L-ext
    n2odict[scens[5]]=[[2300],[np.nan]]  #    VL-ext
    n2odict[scens[6]]=[[2100,2200],[np.nan,2]] #    L-ext-OS

    # %%
    sulfdict={}
    sulfdict[scens[0]]=[[2300],[np.nan]] # H-ext
    sulfdict[scens[1]]=[[2100,2150],[np.nan,10]] # H-ext-OS
    sulfdict[scens[2]]=[[2300],[np.nan]] #    M-ext
    sulfdict[scens[3]]=[[2040,2050,2100],[np.nan,24,10]]     #    MOS-ext
    sulfdict[scens[4]]=[[2300],[np.nan]]   #    L-ext
    sulfdict[scens[5]]=[[2300],[np.nan]]  #    VL-ext
    sulfdict[scens[6]]=[[2025,2030,2040,2050,2100],[np.nan,60,45,24,10]] #    L-ext-OS

    # %%
    f.fill_from_csv(
        emissions_file='../data/emissions/extensions_1750-2500.csv',
        forcing_file='../data/forcing/volcanic_solar.csv',
    )

    # %%
    f.emissions.sel(scenario='high-extension').loc[:]=f.emissions.sel(scenario='high-overshoot').copy()

    # %%

    for s in scens:
        corescen=f.emissions.sel(scenario=s, specie='CH4',config=1234).where(f.emissions.timepoints<=ch4dict[s][0][0], drop=True)
        histtim=f.emissions.timepoints.where(f.emissions.timepoints<=ch4dict[s][0][0], drop=True)
        tmpf=f.emissions.sel(scenario=s, specie='CH4',config=1234).copy()
        tmpf[:]=(f.emissions.timepoints,np.interp(f.emissions.timepoints,np.hstack([histtim,ch4dict[s][0][1:]]),np.hstack([corescen,ch4dict[s][1][1:]])))[1]
        f.emissions.sel(scenario=s, specie='CH4').loc[:,:]=tmpf
    for s in scens:
        corescen=f.emissions.sel(scenario=s, specie='N2O',config=1234).where(f.emissions.timepoints<=n2odict[s][0][0], drop=True)
        histtim=f.emissions.timepoints.where(f.emissions.timepoints<=n2odict[s][0][0], drop=True)
        tmpf=f.emissions.sel(scenario=s, specie='N2O',config=1234).copy()
        tmpf[:]=(f.emissions.timepoints,np.interp(f.emissions.timepoints,np.hstack([histtim,n2odict[s][0][1:]]),np.hstack([corescen,n2odict[s][1][1:]])))[1]
        f.emissions.sel(scenario=s, specie='N2O').loc[:,:]=tmpf
    for s in scens:
        corescen=f.emissions.sel(scenario=s, specie='Sulfur',config=1234).where(f.emissions.timepoints<=sulfdict[s][0][0], drop=True)
        histtim=f.emissions.timepoints.where(f.emissions.timepoints<=sulfdict[s][0][0], drop=True)
        tmpf=f.emissions.sel(scenario=s, specie='Sulfur',config=1234).copy()
        tmpf[:]=(f.emissions.timepoints,np.interp(f.emissions.timepoints,np.hstack([histtim,sulfdict[s][0][1:]]),np.hstack([corescen,sulfdict[s][1][1:]])))[1]
        f.emissions.sel(scenario=s, specie='Sulfur').loc[:,:]=tmpf

    # %%
    gwp_nonco2=gwpmat.copy()
    gwp_nonco2.loc['CO2 AFOLU']=np.nan
    gwp_nonco2.loc['CO2 FFI']=np.nan


    # %%
    nonco2=f.emissions.sel(specie='CO2 FFI')[:,:,0].copy()
    for specie in f.emissions.specie.values:
        try: 
            gwp=gwp_nonco2[specie]
        except:
            gwp=np.nan
        if ~np.isnan(gwp):
            nonco2=nonco2+f.emissions.sel(specie=specie)[:,:,0]*gwp
        else:
            0  


    # %%
    ncflr=np.ones(len(scens))
    for i in range(len(scens)):
        ncflr[i]=nonco2.sel(scenario=scens[i])[-1]/1e6
    ncflr

    # %%
    f.emissions.sel(scenario='high-overshoot', specie='CO2',config=1234,timepoints=np.array([2030,2040,2050,2075,2100,2125])+.5)

    # %%


    # %%


    # %%
    sdict={}
    sdict[scens[0]]=[[2025,2030,2040,2050,2075,2100,2175,2275],[np.nan,43, 50, 57.5, 68 , 74,
        74,0]] # H-ext
    sdict[scens[1]]=[[2025,2030,2040,2050,2075,2100,2160,2200,2350,2400],[np.nan,43, 50, 57.5, 68 , 74,
        0,-34,-34,0]]# H-ext-OS
    #sdict[scens[2]]=[[2125,2200],[np.NaN,0]]# M-ext
    sdict[scens[2]]=[[2025,2030,2100,2200],[np.nan,41,41,0]]# M-ext
    #sdict[scens[3]]=[[2025,2030,2040,2050,2074,2110,2250,2275],[np.NaN,44,44,10,-3,-13,-13,-ncflr[3]]]# MOS-ext

    sdict[scens[3]]=[[2025,2030,2040,2050,2060,2100,2125,2275,2300],[np.nan,41,41,38,32,0,-13,-13,0]]# MOS-ext
    #sdict[scens[4]]=[[2025,2030,2040,2050,2080,2100,2240,2290],[np.nan,40,35,25,0,-7,-7,0]]# L-ext
    sdict[scens[4]]=[[2025,2030,2040,2050,2080,2100,2240,2290],[np.nan,42.5,38,22,-1,-7,-7,0]]# L-ext
    sdict[scens[5]]=[[2025,2030,2040,2050,2060,2070,2080,2090,2100,2125,2225,2275,2300],[np.nan,37.5,15,2,-3,-4.5,-5,-5,-5,-5,-5,-5,0]]# VL-ext

    #sdict[scens[6]]=[[2125,2150,2250,2275],[np.NaN,-25,-25,0]]# L-ext-OS
    sdict[scens[6]]=[[2025,2030,2040,2050,2065,2100,2150,2200,2250],[np.nan,42.5,37.5,6,-8,-24,-30,-30,0]]# L-ext-OS


    # %%

    for s in scens:
        corescen=(f.emissions.sel(scenario=s, specie='CO2 FFI',config=1234)+f.emissions.sel(scenario=s, specie='CO2 AFOLU',config=1234)).where(f.emissions.timepoints<=sdict[s][0][0], drop=True)
        corelu=f.emissions.sel(scenario=s, specie='CO2 AFOLU',config=1234)
        histtim=f.emissions.timepoints.where(f.emissions.timepoints<=sdict[s][0][0], drop=True)
        tmpf=f.emissions.sel(scenario=s, specie='CO2 FFI',config=1234).copy()
        tmpf[:]=(f.emissions.timepoints,np.interp(f.emissions.timepoints,np.hstack([histtim,sdict[s][0][1:]]),np.hstack([corescen,sdict[s][1][1:]])))[1]
        
        f.emissions.sel(scenario=s, specie='CO2 FFI').loc[:,:]=tmpf-corelu

    # %% [markdown]
    # blending for 4 years, starting with 2022

    # %%


    # %%
    cyrs=4
    cfx=2022.5
    bstrt=f.emissions.sel(timepoints=cfx).copy()

    for i in np.arange(0,cyrs):
        yr=cfx+i
        f.emissions.loc[dict(timepoints=yr)]=bstrt

    byrs=5
    bfx=2025.5
    for i in np.arange(0,byrs):
        yr=bfx+i
        e_org=f.emissions.loc[dict(timepoints=yr)].copy()
        f.emissions.loc[dict(timepoints=yr)]=bstrt*(1-i/byrs)+e_org*(i/byrs)

    f.fill_species_configs('../data/fair-parameters/species_configs_properties_1.4.1.csv')
    f.override_defaults('../data/fair-parameters/calibrated_constrained_parameters_1.4.1.csv')

    
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)
    return f
