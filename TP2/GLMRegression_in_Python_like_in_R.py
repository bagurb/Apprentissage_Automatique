import pandas as pd 
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

def signifCode(x):
    if x < 0.001:
        r = '***'
    elif x < 0.01:
        r = '**'
    elif x < 0.05:
        r = '*'
    elif x < 0.1:
        r = '.'
    else:
        r = ''
    return r 
    
def printSignifCodes(res):
    """Affiche les codes significatifs, à la R
    pour un modèle statistique res"""
    import pandas as pd 
    S = pd.DataFrame(pd.concat( (res.pvalues, res.pvalues.map(signifCode)), axis=1 ))
    S.columns = ['p-value', 'Code']
    print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    print(S)
    return S


def GLMsummary(res):
    from IPython.display import HTML
    smry  = res.summary()
    S = pd.DataFrame(smry.tables[1])
    S.columns=S.iloc[0]
    #S.index=S.iloc[:,0]
    S = S.drop(0, axis=0)
    S.columns = ['idx', 'coef', 'std err', 'z', 'P>|z|', '[0.025', '0.975]']
    S.index=S.idx
    S = S.drop('idx', axis=1)
    u = pd.DataFrame(pd.concat( (res.pvalues, res.pvalues.map(signifCode)), axis=1 ))
    u.columns = ['p-value', 'Code']
    S.index=u.index
    
    tab1 = pd.concat( (S, u), axis=1 )
    tab0 = smry.tables[0]
    smry.tables[1] = tab1
    
    display(tab0)
    display(tab1)
    display(HTML("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"))
    display(HTML("AIC with {} ddl = {:.2f}".format(res.df_model+1, res.aic) ))
    
def glm_residplot(res, figsize=(8, 5)):
    import scipy as sp
    import seaborn as sns
    import matplotlib.pyplot as plt
        
    def jfb_residplot(x, y, data=None, lowess=False, x_partial=None, y_partial=None,
              order=1, robust=False, dropna=True, label=None, color=None,
              scatter_kws=None, line_kws=None, ax=None):
        """This is taken and adapted from seaborn residplot. 
        This function draw a scatterplot. You can
        optionally fit a lowess smoother to the residual plot, which can
        help in determining if there is structure to the residuals.
        """

        from seaborn.regression import _RegressionPlotter
        plotter = _RegressionPlotter(x, y, data, ci=None,
                                     order=order, robust=robust,
                                     x_partial=x_partial, y_partial=y_partial,
                                     dropna=dropna, color=color, label=label)
        if ax is None:
            ax = plt.gca()

            # Set the regression option on the plotter
        if lowess:
            plotter.lowess = True
        else:
            plotter.fit_reg = False

        # Plot a horizontal line at 0
        ax.axhline(0, ls=":", c=".2")

        # Draw the scatterplot
        scatter_kws = {} if scatter_kws is None else scatter_kws.copy()
        line_kws = {} if line_kws is None else line_kws.copy()
        plotter.plot(ax, scatter_kws, line_kws)
        return ax


    def _glm_residplot(x,y,res,title="", xlabel="", ylabel="",n=3, top_n=None):
        f, ax = plt.subplots(figsize=figsize)
        ax = jfb_residplot(x=x, y=y, lowess=True, 
                line_kws={'color': 'red','lw': 1,'alpha': 0.8}, ax=ax)
        if top_n is None:
            top_n = np.flip(np.argsort(np.abs(y)), 0)[:n]
        ax.set_ylim(np.array(ax.get_ylim())*1.1)
        for i in top_n:
            ax.annotate(i, xy=(x[i], y[i]))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    # Pearson residuals vs fitted   
    title = "Pearson residuals vs fitted"
    xlabel = "Predicted values"; ylabel = "Residuals"
    _glm_residplot(res.fittedvalues,res.resid_pearson, res, 
                   title=title, xlabel=xlabel, ylabel=ylabel,n=3)    
    
    # Deviance vs fitted    
    title = "Deviance residuals vs fitted"
    xlabel = "Predicted values"; ylabel = "Residuals"
    _glm_residplot(x=res.fittedvalues,y=res.resid_deviance,res=res, 
                   title=title, xlabel=xlabel, ylabel=ylabel,n=3)    
    # Pearson residuals vs leverage    
        # leverage, from statsmodels internals
    model_leverage = res.get_influence().hat_matrix_diag
        # cook's distance, from statsmodels internals
    model_cooks = res.get_influence().cooks_distance[0]
    top_n = np.flip(np.argsort(model_cooks), 0)[:3]
    title = "Pearson residuals vs leverage"
    xlabel = "Leverage"; ylabel = "Residuals"    
    _glm_residplot(x=model_leverage,y=res.resid_pearson,res=res, 
                   title=title, xlabel=xlabel, ylabel=ylabel,n=3, top_n=top_n)    
    # Q-Q plot
    f, ax = plt.subplots(figsize=figsize)
    sp.stats.probplot(res.resid_pearson, plot=ax, fit=True)
    plt.autoscale(enable=True, axis='both', tight=None)
    ax.set_title('Normal Q-Q')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Residuals')
    
    # Histogramme deviances
    f, ax = plt.subplots(figsize=figsize)
    sns.distplot(res.resid_deviance, kde=True)
    ax.set_title('Deviance histogram')
    ax.set_xlabel('Deviance')
    ax.set_ylabel('Counts')        
    