"""
Author: Amogh Jalihal
Date: 2025-08-24
Commentary:
Preprocesses qPCR data over dilutions to assess viability.
"""
from QLD import quantifyInputFromSerialDilution, PoissonJoint
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Rectangle
from scipy.stats import bootstrap
import pandas as pd
import seaborn as sns
import numpy as np
import sys

def viability_pipeline(viabilitydf : pd.DataFrame,
                       task : str = "initial",
                       initial_viability : pd.DataFrame = None,
                       dilutioncolumn : str  ="dilution",
                       make_report : bool =False,
                       exppref : str ="experiment",
                       foldDilution: float=10.,
                       lower_ct_cutoff : float = 10.,
                       upper_ct_cutoff : float = 40.,
                       groupings : list= [] ):
    """
    TODO finish documentation
    viabilitydf is a DataFrame with replicate group, replicate, dilution, 
    and CT for each well, along with optional columns passed in the =groupings= argument.
    This function operates in two 'task modes':
    1. "initial" mode assumes the first dilution to start computing dispersion metrics is 1.
    2. "assessment" mode takes in an additional DataFrame,
       initial_viability, which has the following columns:
       1. "growth" : 
       assumes the first dilution to start computing dispersion metrics is 1.
    returns
    - [] if no viable strains initially
    - [d1, d2] list of 2 integers if initially viable, where d1 and d2
      are the first and second dilutions that need to be evaluated 
      for 1% survival
    - [d2] list of one integer, when viability is low enough that 
      only 10-100% survival can be assessed.
    """

    ### Normalize column names
    if "dilution" not in viabilitydf.columns:
        viabilitydf = viabilitydf.rename({dilutioncolumn:"dilution"},
                                         axis="columns")

    _required_columns = ["replicate","dilution","CT","Sp", "Amp Status"]
    
    ### Ensure required columns are present
    for req_c in _required_columns:
        try: 
            assert req_c in  list(viabilitydf.columns)
        except Exception:
            print(viabilitydf.columns)
            print('ERROR: ', req_c, "which is a required column, is not found in dataframe!")
            raise
    returnlist = ["repid","dilution","replicate",
                  "notes","d1","d2","lci","uci","var",
                  "growthcount","CT_mean","CT_std",
                  "is_valid_ct", "growth"]
    if len(groupings) > 0:
        ### Ensure required columns are present
        for grp_c in groupings:
            try: 
                assert grp_c in  viabilitydf.columns
            except AssertionError:
                print(grp_c, "which is a grouping column that you have supplied, is not found in dataframe!")
                raise
        groupnames = "_".join([str(viabilitydf[g].values[0]) for g in groupings if g != "Sp" ])
    else:
        groupnames = ""
    axprune, axfill = None, None
    if make_report:
        ### Initialize variables for report generation
        plt.close("all")
        fig = plt.figure(figsize=(15,10))
        
        mosaic = """
        AAA
        BCD
        """
        axes = fig.subplot_mosaic(mosaic)
        axprune = axes["A"]
        axfill = [axes["B"], axes["C"], axes["D"]]
        directory = groupnames
        if not os.path.exists(f"img/{exppref}/"):
            os.makedirs(f"img/{exppref}/")            
        if not os.path.exists(f"img/{exppref}/{directory}"):
            os.makedirs(f"img/{exppref}/{directory}/", exist_ok=False)

    ### 0. Normalize column names
    sp = viabilitydf.Sp.unique()[0]
    print("------- Species", sp)

    if task == "assessment":
        initviab_grouping_cols = ["growth","d1","d2"]
        groupcols = []
        for col in groupings:
            if col in initial_viability.columns:
                initial_viability = initial_viability[initial_viability[col] == viabilitydf[col].values[0]]
        initial_viability = initial_viability[initial_viability.Sp ==\
                                              sp][["growth","d1","d2"]].drop_duplicates()

        if initial_viability.shape[0] == 0 :
            print("No initial viability")
            return(viabilitydf.assign(is_valid_ct=0,CT_mean=np.nan,CT_std=np.nan,growthcount=0,
                                      growth=0, 
                                      notes="Initially inviable",
                                      d1 = 0,
                                      d2 = 0,
                                      groupings=groupnames,
                                      var = 0,
                                      lci= 0,
                                      uci=0)\
                   [returnlist].drop_duplicates())

        """
        Order matters!!
        Replace d2 first, then d1.
        Check 1:
        If d1 = 0 and d2 > 0 (i.e. d2 = 1), set dilution 1 to d2.
        """
        if np.isnan(initial_viability["d1"].values[0]) or (initial_viability["d1"].values[0] == 0.):
            viabilitydf["d1"] = 0
            viabilitydf.loc[viabilitydf.dilution == 1, "dilution"] = 0
            if np.isnan(initial_viability["d2"].values[0]) or (initial_viability["d2"].values[0] == 0.0):
                viabilitydf["d2"] = 0
                viabilitydf.loc[viabilitydf.dilution == 2, "dilution"] = 0
            else:
                viabilitydf["d2"] = int(initial_viability.d2.unique()[0])
                viabilitydf.loc[viabilitydf.dilution == 2, "dilution"] = int(viabilitydf.d2.values[0])
        else:
            viabilitydf["d1"] = int(initial_viability["d1"].values[0])
            viabilitydf["d2"] = int(initial_viability.d2.unique()[0])
            viabilitydf.loc[viabilitydf.dilution == 2, "dilution"] = int(initial_viability.d2.unique()[0])
            viabilitydf.loc[viabilitydf.dilution == 1, "dilution"] = int(initial_viability.d1.unique()[0])

        try:
            assert viabilitydf[viabilitydf.dilution > 0].shape[0] > 0
        except AssertionError:
            print("Invalid initial viability dilution selection. No dilutions left on which to operate!")
            return(viabilitydf.assign(is_valid_ct=0,CT_mean=np.nan,CT_std=np.nan,growthcount=0,
                                      growth=0, 
                                      notes="No dilutions to assess. Either initial viability error, or no valid measurements were performed",
                                      d1 = 0,
                                      d2 = 0,
                                      groupings=groupnames,
                                      var = 0,
                                      lci= 0,
                                      uci=0)\
                   [returnlist].drop_duplicates())

    ### 1. Assess initially viable strains
    has_growth_p, growthcountdf = has_growth(viabilitydf, 
                                             lower_ct_cutoff = lower_ct_cutoff,
                                             upper_ct_cutoff = upper_ct_cutoff, )
    ### -- Adds columns = 'growthcount', 'CT_mean', and 'CT_std'
    cols = []
    cols = list(groupings)
    cols.extend(["dilution","repid","replicate","CT","Amp Status"])

    analysisdf = viabilitydf.merge(growthcountdf, on=["dilution"])
    print("... finished pruning no growths")
    if not has_growth_p:
        print("LOG: No growth identified, returning null template.")
        return(viabilitydf.assign(is_valid_ct=0,CT_mean=np.nan,CT_std=np.nan,growthcount=0,
                                      growth=0, 
                                  notes="No initial viability",
                                  d1 = np.nan,
                                  d2 = np.nan,
                                  var = np.nan,
                                  groupings="_".join(groupnames),
                                  lci=np.nan,
                                  uci=np.nan)\
               [returnlist].drop_duplicates())
    
    ### 2. Prune noise
    pruned = prune_noisy_measurements(viabilitydf, growthcountdf, 
                                      debug = make_report,
                                      lower_ct_cutoff = lower_ct_cutoff,
                                      upper_ct_cutoff = upper_ct_cutoff,
                                      task = task,
                                      initial_viability = initial_viability,
                                      ax=axprune)
    print(pruned.shape)
    print("... pruned noisy measurements")
    ### -- Adds columns = 'is_valid_vt'
    analysisdf = analysisdf.merge(pruned,\
                                on=list(set(analysisdf.columns)\
                                        .intersection(set(pruned.columns))))
    print(analysisdf.shape)
    ### 3. Fill dropouts and assess viability
    ## Fill back in flase dropouts from previous step
    try:
        mle, lci, uci, var, filled = fill_dropouts_and_assess_viability(pruned, foldDilution, 
                                                   debug=make_report,task=task,
                                                   axes=axfill)
        print(mle)
    except:
        print("ERROR: fill_dropouts_and_assess_viability() failed.")
        raise
    ### -- Adds columns = 'growth'
    print(filled)
    analysisdf = analysisdf.merge(filled, 
                                  on=list(set(analysisdf.columns)\
                                          .intersection(set(filled.columns))))
    d1, d2 = calculate_dilutions_for_viab_estimation(mle, foldDilution)
    if make_report:
        fig.suptitle(sp)
        plt.tight_layout()
        plt.savefig(f"img/{exppref}/{directory}/{sp}.png")
        plt.close("all")
    returndf = analysisdf.assign(growth=round(mle, 2), 
                             d1 = int(d1),
                             d2 = int(d2),
                             lci= round(lci,2), 
                             uci=round(uci,2), 
                             groupings="_".join(groupnames),
                             var=round(var,2))\
               [returnlist].drop_duplicates()
    return(returndf)


def calculate_dilutions_for_viab_estimation(mle, foldDilution):
    if (not np.isfinite(mle)) or np.isnan(mle) or (mle == 0):
        return(0,0)
    growth_order_magnitude = round(np.emath.logn(foldDilution, mle))
    d1,d2 = growth_order_magnitude-1, growth_order_magnitude
    return((d1, d2))

def fill_dropouts_and_assess_viability(df, foldDilution, debug = False, 
                                       axes= None, task="initial"):
    """
    If task is "assessment", we make a strong assumption that the dilution succeeding the last dilution has 
    _zero_ growth. This is necessary to estimate a lower bound on the viability. 
    """
    df = df.assign(group_rep = df.repid.astype(str).str.cat(df.replicate.astype(str),sep="_"))
    df.loc[df.CT.isna(),"CT"] = 45
    #df.loc[df.CT > 30, "is_valid_ct"] = False
    thresholdCT = df[df.is_valid_ct].CT.median() 
    df = df.assign(growth = 0.)
    df.loc[df.is_valid_ct, "growth"] = 1.
    ### HAndle initial viability experiments here.
    df = df[df.dilution > 0 ]

    ## Lower bound estimates for assessment task
    lastdildf = df[df.dilution == df.dilution.max()]
    lastdildf["dilution"] =lastdildf["dilution"] + 1
    lastdildf["growth"] = 0 
    lastdildf["CT"] = 45
    df = pd.concat([df,lastdildf]).reset_index(drop=True)
    # if task == "assessment":
    #     lastdildf = df[df.dilution == df.dilution.max()]
    #     lastdildf["dilution"] =lastdildf["dilution"] + 1
    #     lastdildf["growth"] = 0 
    #     lastdildf["CT"] = 45
    #     df = pd.concat([df,lastdildf]).reset_index(drop=True)

    ## For visualization
    try:
        plate = df.pivot(index="group_rep",columns="dilution", values="CT")
    except:
        print(df.sort_values(by="group_rep"))
        raise
    
    
    sp = df.Sp.unique()[0]
    growth = df.pivot(index="group_rep",columns="dilution", values="growth")
    ## Check if there is any growth at all:
    sumgrowth = 0
    for i, row in growth.iterrows():
        sumgrowth += row.sum()
    if sumgrowth == 0 : 
        return(0,0,0,0,df)
        

    mle_prefill, lci ,uci , var = quantifyInputFromSerialDilution(growth, foldDilution=foldDilution)


    ## Clean false positives: any singleton growth in a group with two preceding dilutions with no growth is a false positive
    ## TODO
    print("In fill_drop -- 1.2: ", df)
    for dilution in range(df.dilution.max(),max(df.dilution.min(),1),-1) :
        for g in df.repid.unique():
            subdf = df[(df.dilution == dilution) & (df.repid == g)]
            prevdf = df[(df.dilution == dilution-1) & (df.repid == g)]
            prevprevdf = df[(df.dilution == dilution-2) & (df.repid == g)]
            if prevdf.growth.sum() == 0:
                if prevprevdf.growth.sum() == 0:
                    df.loc[subdf.index, "growth"] = 0

                 
    ## Next, if there is growth till dilution d, fill everything until dilution d-2
    countdf = df.groupby("dilution")\
                .apply(lambda gdf: gdf.assign(gcount = gdf[gdf.growth > 0.].shape[0])[["gcount"]]\
                       .drop_duplicates())\
                .reset_index()

    ## EXPERIMENTAL
    ##  If an entire repid is completeley empty, remove these reps.
    for ri in df.repid.unique():
        hasgrowth = False
        for i, row in df[df.repid == ri].iterrows():
            if row.growth == 1:
                hasgrowth = True
                break
        if not hasgrowth:
            df = df[df.repid != ri]
                
    # For everything else, treat no growth as qPCR dropouts and back fill them.
    maxgrowthdil = countdf[countdf.gcount > 1].dilution.max()
    df.loc[df.dilution <= maxgrowthdil -1, "growth"] = 1
    print("In fill_drop -- 1.5: ", df)

    growthfilled = df.pivot(index="group_rep",columns="dilution", values="growth")
    print("In fill_drop -- 2: ", growthfilled)
    mle_postfill, lci ,uci , var = quantifyInputFromSerialDilution(growthfilled, foldDilution=10.15)


    if debug:
        ax1, ax2, ax3 = axes
        ax1.set_title(f"CT")
        sns.heatmap(plate, vmin=thresholdCT-5,
                    vmax=35,linewidth=1,
                    cmap="viridis",ax=ax1)
        # ax = fig.add_subplot(2,2,3)
        ax2.set_title(f"Inferred Growth: Estimate ={round(mle_prefill)}")
        sns.heatmap(growth, vmin=0,vmax=1,linewidth=1,linecolor="k",
                    cmap="gray",ax=ax2)

        ax3.set_title(f"Back-filled: Estimate = {round(mle_postfill)}")
        sns.heatmap(growthfilled, vmin=0,vmax=1,linewidth=1,linecolor="k",
                    cmap="gray",ax=ax3)
    return(mle_postfill, lci, uci, var, df)

def plot_scatter(df, ax):
    g = sns.scatterplot(data=df, x="dilution",y="CT", 
                        style="is_low_dispersion", 
                        hue="is_valid_ct",hue_order=[True,False],
                        size="repid",#alpha=0.5,
                        palette="tab10",ax=ax)
    g.set(ylim=[0,40],xlim=[0,8])    
    g.legend(framealpha=0, fontsize=12)

def prune_noisy_measurements(fulldf, growthcountdf, 
                             lower_ct_cutoff = 10.,
                             upper_ct_cutoff = 40.,
                             debug = False, ax=None,
                             task = "initial", 
                             initial_viability = None):
    """
    Use rep group information to identify the dispersion range starting from the lowest dilution
    TODO There are a lot of empirically determined CT values. see how robust this function is to these choices
    Returns df with the following columns:
    1. is_valid_ct
    """
    df = fulldf[(fulldf.CT < upper_ct_cutoff) & (fulldf.CT> lower_ct_cutoff)]
    print("Is there anything left after filtering?", df.shape)
    if df.shape[0] == 0:
        fulldf = fulldf.assign(is_valid_ct = False,
                               is_low_dispersion = False,
                               notes = "CT outside valid range")
        plot_scatter(fulldf, ax)
        return(fulldf)

    collect = []
    sp = fulldf.Sp.unique()[0]
    ### Compute dispersion metrics for each group

    first_dilution = 1
    if task == "assessment":
        d1, d2 = df.d1.unique()[0], df.d2.unique()[0]
        if d1 > 0:
            first_dilution = d1
            print("USING d1=", d1)
        elif d2 > 0:
            first_dilution = d2
            print("USING d2=", d2)
        else:
            print("WARNING: INITIAL DILUTION UNDEFINED. PROCEEDING WITH D=1.")
    df = df[df.dilution >= first_dilution]
    try:
        assert df.shape[0] > 0
    except AssertionError:
        print("")
    stats = df.groupby(["dilution","repid"])\
               .CT.agg({"mean","median","std","count","min","max"})\
                .reset_index().drop_duplicates()

    ### Compute the range of CTs for each group
    stats  = stats.assign(ct_range = stats["max"] - stats["min"])
    
    ### Each dilution can have a slightly different range. 
    ### First, flag the low dispersion rep groups in each dilution. Use CT_RANGE as a reasonable upper limit of spread
    CT_RANGE = 10 # 6
    stats = stats.assign(is_low_dispersion = False)

    stats.loc[stats.ct_range < CT_RANGE, "is_low_dispersion"] = True

    # ### Check if the entire dataset is high noise, crash early and return Nans for CTs.
    # if stats[stats.is_low_dispersion].shape[0] == stats.shape[0]:
    #     df = df.assign(is_valid_ct = False)
    #     df["CT" ] = 45
    #     # df.loc[df.is_valid_ct.isna(), "is_valid_ct"] = False
    #     # df.loc[df.CT.isna(), "CT"] = float(45.)
    #     return(df)

    ### Each rep group can have systematic behavior that makes 
    ### transferring statistics between groups pointless
    ### All CTs within a "low dispersion group" should be 
    ### considered valid IF the mean CT of this rep group 
    ### isn't more than 4 CTs from that of the other rep groups
    stats = stats.assign(is_lowdisp_repgroup = False)
    stats = stats.merge(stats[stats.is_low_dispersion].groupby("dilution")["median"].median(),\
                        on="dilution",suffixes=["","_aggregate"], how="left")
    stats.loc[stats["median"] < (stats.median_aggregate + 4), "is_lowdisp_repgroup"] = True
    stats.loc[~stats.is_low_dispersion, "is_lowdisp_repgroup"] = False

    ### So far we have made decisions about rep groups _within_ dilutions.
    ### As we get to the end of the dilution series, we might be left with a single rep group,
    ### and this might not have a valid CT range. We now compare to the previous dilution, and remove
    ### repgroups with aggregate_medians more than 6 from that of the previous dil.
    ## Merge these statistics back in the dataframe
    df = df.merge(stats, on =["dilution","repid"])
    df = df.assign(is_valid_ct = False)

    df.loc[(df.dilution < first_dilution) , "is_valid_ct"] = False
    df.loc[(df.dilution == first_dilution) &( df.is_lowdisp_repgroup), "is_valid_ct"] = True
    ## First pass
    ## Check if 
    for d in range(first_dilution, int(stats.dilution.max()+1)):
        print("pruning dilution", d)
        valid_cts_df = df[(df.dilution <=d) & (df.is_valid_ct)]
        cum_agg_range_max,cum_agg_range_min  = valid_cts_df.CT.max(), valid_cts_df.CT.min()
        df.loc[(df.dilution == d)\
                & ((df.CT > (cum_agg_range_min-3))\
                   & (df.CT < (cum_agg_range_max+1.5))),
                "is_valid_ct"] = True
    ### Conservative second pass
    valid_cts_df = df[(df.is_valid_ct)]
    cum_agg_range_max,cum_agg_range_min  = valid_cts_df.CT.max(), valid_cts_df.CT.min()
    df.loc[((df.CT > (cum_agg_range_min))\
             & (df.CT < (cum_agg_range_max))),
            "is_valid_ct"] = True
    if debug:
        plot_scatter(df, ax)

    df = df.merge(fulldf[["Sp","dilution","repid","replicate","CT"]],
                  on=["dilution","repid","replicate","Sp"], how="right",suffixes=["","_old"])
    df.loc[df.is_valid_ct.isna(), "is_valid_ct"] = False
    df.loc[df.CT.isna(), "CT"] = float(45.)
    df["notes"] = ["Low dispersion. "
                   if row.is_low_dispersion
                   else
                   "High dispersion. "
                   for i, row in df.iterrows()]

    return(df)

def has_growth(df, lower_ct_cutoff : float = 10., upper_ct_cutoff = 40):
    """
    Start at the highest dilution, decrement dilution until all replicates have non zero measured growth.
    If no well has growth, return False, else return True.
    params:
    - lower_ct_cutoff : automatically filter out CTs below this cutoff. can be set to 0.
    - upper_ct_cutoff : Max CT below which growth is assumed to be real.
    returns:
    - Growth status: boolean
    - Growth df: Dataframe that records the count of wells scored as having growth in each dilution, 
      along with statistics of mean and std of the CT at each well. Optional object, can be used to
      compute and refine metrics if needed.
    """
    try:
        for required_col in ["repid","replicate","dilution"]:
            assert required_col in df.columns
    except AssertionError:
        print(df.Sp.unique()[0], ": Dataframe missing one of repid, replicate, or dilution columns. growth assessment failed.")
        raise Exception
                
    growthmap = []
    df = df[df.CT > lower_ct_cutoff]
    df = df.assign(grp_rep = df.repid.astype(str).str.cat(df.replicate.astype(str), sep="_"),
                   growth = 0)
    df.loc[df.CT< upper_ct_cutoff, "growth"] = 1
    try:
        plate = df.pivot(index="grp_rep",
                         columns="dilution",
                         values="growth")
    except:
        print("ERROR: has_growth()")
        print(df)
        raise
    mle, lci ,uci , var = quantifyInputFromSerialDilution(plate, foldDilution=10.15)

    for dilution in sorted(df.dilution.unique()):
        """
        If a layout is provided to the DA2 software, the CT corresponding to an UNDETERMINED
        signal is less than 45 (or the max number of cycles run). The softare reports this
        as NO_AMP. 
        Use this status to filter out signal
        """
        dfoi = df[(df['Amp Status']!="NO_AMP") & (df.dilution == dilution) & (df.CT < 44)]
        if dfoi.shape[0]== 0:
            growthcount = 0
        else:
            growthcount = dfoi.CT.count()
        CT_mean = round(dfoi.CT.mean(),1)
        CT_std = round(dfoi.CT.std(),1)
        growthmap.append({"dilution":dilution,
                          "growthcount":growthcount, 
                          "CT_mean":CT_mean,
                          "CT_std":CT_std})
    growthdf = pd.DataFrame(growthmap ).sort_values(by="dilution").reset_index(drop=True)
    if (mle  < 1) :
        return(False, growthdf)
    else:
        return(True,growthdf)
