
# coding: utf-8

# # Utilisation de Statsmodels pour faire une régression linéaire

import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf

# # Émulation sous Python de la régression linéaire R

# ## Fontions utilitaires

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


def RsummaryTable(est):
    import pandas as pd
    T = pd.DataFrame(
        columns=['Estimate', 'Std. Error', 't value', 'Pr(>|t|)', 'code'])
    T['Estimate'] = est.params
    T['Std. Error'] = est.bse
    T['t value'] = est.tvalues
    T['Pr(>|t|)'] = est.pvalues
    T['code'] = est.pvalues.map(signifCode)
    return T


def Rsummary(est):
    from numpy import sqrt, sum
    import io
    f = io.StringIO()
    print("Call\nsmf.ols('{}', data=...)".format(est.model.formula), file=f)
    print("\nCoefficients:",file=f)
    T = RsummaryTable(est)
    print(T, file=f)

    q = est.resid.quantile([0, 0.25, 0.5, 0.75, 1])
    q.index = ['Min', '1Q', 'Median', '3Q', 'Max']
    print("\nResiduals:", file=f)
    print(q.to_frame().T, file=f)
    print('---', file=f)
    print("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1", file=f)
    texte = """
Residual standard error: {:.4f} on {:n} degrees of freedom
Multiple R-squared:  {:.4f},	Adjusted R-squared:  {:.3f} 
F-statistic: {:.3f} on {:n} and {:n} DF,  p-value: {:.3e}
""".format(sqrt(sum(est.resid**2)/est.df_resid),
           est.df_resid, est.rsquared, est.rsquared_adj, est.fvalue,
           est.df_model, est.df_resid, est.f_pvalue)
    print(texte, file=f)
    return f.getvalue()

def summary(est):
    import statsmodels
    if isinstance(est, statsmodels.regression.linear_model.RegressionResultsWrapper):    
        print(Rsummary(est))
    else:
        print("TypeError")

def Rplot(est, figsize=(8, 5)):
        import scipy as sp
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        response = est.model.endog_names
        sns.residplot(
            est.fittedvalues,
            response,
            data=est.model.data.frame,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red',
                      'lw': 1,
                      'alpha': 0.8},
            dropna=False,
            ax=ax)
        ax.set_title('Residuals vs Fitted')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')

        f, ax = plt.subplots(figsize=figsize)
        sp.stats.probplot(est.resid, plot=ax, fit=True)
        plt.autoscale(enable=True, axis='both', tight=None)
        ax.set_title('Normal Q-Q')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')

def update_formula(formula, data):
    "This is because . is not interpreted in patsys/statsmodels formulas"
    import re
    all_columns = "+".join(data.columns)
    resp = formula.split('~')[0].strip()
    #out = re.sub('[\+\s]*'+resp+'[\s\+]*', '', all_columns)
    out = re.sub(resp+'[\s]*\+', '', all_columns)
    return formula.replace('.', out)

def vif(data, intercept=True):
    """
    vif(data)
    Compute the vif for data 
       - adds an intercept if intercept=True
       - you may remove the response before the call using something 
       like data.drop('response, axis=1)
    Inspired by https://etav.github.io/python/vif_factor_python.html with small adaptations
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from numpy import ones
    data = data.copy()
    data.dropna()
    data = data._get_numeric_data() #drop non-numeric cols
    if intercept: data['intercept']=ones(data.shape[0])
    vif = pd.DataFrame(columns=['VIF'])
    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif.index = data.columns
    vif = vif.drop('intercept', axis=0)
    return vif
        
# ## Émulation de la fonction lm (1)

def lm(formula, data):
    """
    Usage: lm(formula, data)
    Example:
    model = lm('Sales ~ TV + Radio', advertising)
    model.summary()
    model.plot()
    """
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    def summary_():
        print(Rsummary(est))

    def plot_():
        Rplot(est)
    
    original_formula = formula
    if '.' in formula: 
        formula = update_formula(formula, data)
    est = smf.ols(formula, data).fit()
    # Add two methods to the est object
    est.summary = summary_
    est.plot = plot_
    # and an attribute
    est.residuals = est.resid
    if '.' in original_formula: est.model.formula = original_formula
        
    return est



# ## Émulation de la fonction lm (2)
# Par une approche objet

import statsmodels
class LinearRegression(statsmodels.regression.linear_model.OLS):
    """
    LinearRegression class after statsmodels', using formula and including
    R-like summary and plot. 
    """
     
    def __init__(self, formula, data):
        import statsmodels.api as sm
        self.model = sm.OLS.from_formula(formula, data)
        self.fitted = False

    def fit(self):
        self.fitted = self.model.fit()
        self.fitted.plot = self.plot
        self.fitted.summary = self.summary
        return self.fitted

    def summary(self):
        if self.fitted: print(Rsummary(self.fitted))

    def plot(self, figsize=(8, 5)):
        if self.fitted:
            import scipy as sp
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=figsize)
            response = self.model.endog_names
            sns.residplot(
                self.fitted.fittedvalues,
                response,
                data=self.model.data.frame,
                lowess=True,
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red',
                          'lw': 1,
                          'alpha': 0.8},
                ax=ax)
            ax.set_title('Residuals vs Fitted')
            ax.set_xlabel('Fitted values')
            ax.set_ylabel('Residuals')

            f, ax = plt.subplots(figsize=figsize)
            sp.stats.probplot(self.fitted.resid, plot=ax, fit=True)
            plt.autoscale(enable=True, axis='both', tight=None)
            ax.set_title('Normal Q-Q')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Standardized Residuals')


def lm2(formula, data):
    """
    Usage: 
    >> z = lm2('Sales ~ Radio+TV', advertising)
    >> z.summary() # to get summary
    >> z.plot() # to get diagnostic plots
    """
    original_formula = formula
    if '.' in formula: formula = update_formula(formula, data)
    model = LinearRegression(formula, data)
    model_fitted = model.fit()
    if '.' in original_formula: est.model.formula = original_formula
    return model


# # **Autre utilisation possible**
#rrr = LinearRegression('Sales ~ Radio+TV', advertising)
#zz = rrr.fit()
#rrr.plot() # or zz.plot()

