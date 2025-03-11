
def make_params(n=100):
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


    pmat=[]
    for i in range(n):
        p={}
        p['tcrecs']=tcrecs[i]
        p['F_scale']=F_scale[i,:]
        p['r0']=r0[i]
        p['rc']=rc[i]
        p['rt']=rt[i]
        pmat.append(p)
    return pmat
