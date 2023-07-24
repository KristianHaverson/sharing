import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, deconvolve
import scipy.stats as stats
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.gridspec as gridspec
import scipy as sp
import matplotlib.pyplot as plt
import iminuit
import iminuit.cost
from scipy.optimize import curve_fit



# Function to stylize the plots
def stylise_plot(ax):
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)


def check_length(arr1, arr2):
    return len(arr1) == len(arr2)


def chi2pdof(arr1, arr2):
    if check_length(arr1, arr2):
        chi_squared = np.sum((arr1 - arr2) ** 2 / (0.00001+arr2))
        dof = len(arr1) - 1
        chi_squared_per_dof = chi_squared / dof
        return chi_squared_per_dof
    else:
        raise ValueError("Arrays must have the same length.")



def fit_angular(X=None,values=None, bins=None, ax=None, a=0, b=0, a_limits=None, b_limits=None, verbose=True):
    
    distribution=angular12_theta

    binned = values is not None and bins is not None and X is None
    if verbose:
      if binned:
        print('Running binned fit')
      else:
        print('Running unbinned fit')
    
    
    E1E2_calc=lambda x: (1, np.exp(x))  # E1 kept constant, E2 vraries ( E2/E1 ratio)
    # E2 varies from exp(-10) to exp(10)
    # phi from 0 to pi
    
    pdf=lambda x, a,b: distribution(*E1E2_calc(a), phi=b).pdf(x)
    cdf=lambda x, a,b: distribution(*E1E2_calc(a), phi=b).cdf(x)


    if binned:
      loss=iminuit.cost.BinnedNLL(values, bins, cdf)
    else:
      loss=iminuit.cost.UnbinnedNLL(X, pdf)
      
    if verbose:print(type(loss))

    m = iminuit.Minuit(loss, a=a,b=b)
    
   
    if a_limits is None:
        m.limits["a"] = (-10,10)
    else:
        m.limits["a"] = a_limits


    if b_limits is None:
        m.limits["b"] = (0,np.pi)
    else:
        m.limits["b"] = b_limits

    m.migrad()
    m.hesse()
    m.minos()

    if ax is not None:
        if not binned: values, bins = np.histogram(X, bins=80, range=(0,np.pi))

        ax.set_xlabel('theta [rad]', fontsize=11)
        ax.set_ylabel('counts/bin', fontsize=11)
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)

        light_purple = '#8A3BE1'  # Hexadecimal for light purple


        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        ax.errorbar(bin_centers, values, xerr=(bins[1]-bins[0])/2, yerr=np.sqrt(values), fmt='o', color=light_purple, label='Data', lw=2)


        #ax.step(bin_centers, values, where='mid', color=light_purple, linewidth=2, alpha=0.5, fillstyle='top')
        #ax.fill_between(bin_centers, values, 0, color=light_purple, alpha=0.1)
        centers=np.linspace(0, np.pi)
        y=pdf(centers,*m.values)*get_histogram_integral(bins,values)
        ax.plot(centers, y, label='Fit', color='#444444', linewidth=2, linestyle=':')
        ax.set_xlabel(r' Theta [rad]')
        ax.set_ylabel('Counts/bin')

        if binned:
          chi2ddof=m.fmin.reduced_chi2
        else:
          binned_loss=iminuit.cost.BinnedNLL(values, bins, cdf)
          fmin=binned_loss(*m.values)
          ddof=len(bins)-1
          chi2ddof=fmin/ddof

        
        ratio_v=np.exp(m.values[0])
        ratio_v_err=np.exp(m.values[0])*m.errors[0]
        ratio_label=f'$R_{{(\mathrm{{E2/E1}})}} = {ratio_v:.2E}\pm{ratio_v_err:.2E}$'
        
        fit_info=[ratio_label,
                  f'$\phi_{{12}} = {np.rad2deg(m.values[1]):.2f}\pm{np.rad2deg(m.errors[1]):.2f}$',
                 f'$\chi^{{2}}_{{v}}={chi2ddof:.3f}$']

        loc='upper right' #if ratio_v<1 else 'lower center'
        ax.legend(title='\n'.join(fit_info), fontsize='large', title_fontsize='large',
                 loc = loc)

    return m


############################################################
############################################################

def get_histogram_integral(x,y):
    return (np.diff(x)*y).sum()

############################################################
############################################################
class angular12_theta(sp.stats.rv_continuous):
    "Angular E1/E2 distribution in theta coordinate"

    def __init__(self, E1, E2, phi):
        super().__init__(self,a=0,b=np.pi)
        self.__inverse_cdf = None
        self.E1=E1
        self.E2=E2
        self.phi=phi

    def _pdf(self, x):
        return (1.5*np.sqrt(5*self.E1*self.E2)*\
                np.cos(self.phi)*np.cos(x) + 0.75*self.E1 - 3.75*self.E2*np.sin(x)**2 +\
                3.75*self.E2)*np.sin(x)**3/(self.E1 + self.E2)
    
    def _cdf(self, x):
        return (0.375*np.sqrt(5*self.E1*self.E2)*\
                (1 - np.cos(x)**2)**2*np.cos(self.phi) + 0.25*self.E1*np.cos(x)**3 -\
                0.75*self.E1*np.cos(x) + 0.5*self.E1 - 1.25*self.E2*(1 - np.cos(x)**2)**2*np.cos(x) +\
                2.0*self.E2*np.cos(x)**5 - 3.75*self.E2*np.cos(x)**3 + 1.25*self.E2*np.cos(x) + 0.5*self.E2)/(self.E1 + self.E2)
   
    def _ppf(self,x):
        return self.inverse_cdf()(x)
    def inverse_cdf(self):
        if self.__inverse_cdf is None:
            x=np.linspace(0,np.pi,100000)
            self.__inverse_cdf = sp.interpolate.interp1d(self.cdf(x),x)
        return self.__inverse_cdf

    ############################################################

def conv(x,y,degreesToConv):
    numberOfPoints = len(x)
    rangeOfVals  = x[numberOfPoints-1] - x[0]
    XPerRad =  numberOfPoints/rangeOfVals 
    XtoConv = XPerRad*np.deg2rad(degreesToConv)
    Res_to_SD = lambda fwhm: fwhm / (2 * np.sqrt(2 * np.log(2)))
    SD_X = Res_to_SD(XtoConv)
    kernel = signal.windows.gaussian(numberOfPoints,SD_X)
    convolved_signal = np.convolve(y, kernel, mode='same')
    convolved_signal = convolved_signal / (sum(kernel))

    return convolved_signal
    ############################################################


class angular12_theta_conv(sp.stats.rv_continuous):
    "Angular E1/E2 distribution in theta coordinate"

    def __init__(self, E1, E2, phi):
        super().__init__(self,a=0,b=np.pi)
        self.__inverse_cdf = None
        self.E1=E1
        self.E2=E2
        self.phi=phi


    def _pdf(self, x,degConv):
        y= (1.5*np.sqrt(5*self.E1*self.E2)*\
                np.cos(self.phi)*np.cos(x) + 0.75*self.E1 - 3.75*self.E2*np.sin(x)**2 +\
                3.75*self.E2)*np.sin(x)**3/(self.E1 + self.E2)
        

        y2= conv(x,y,degConv)
        return y2
    
    def _cdf(self, x,degConv):
        y= self._pdf(x,degConv)
        convolved_pdf_cdf = np.cumsum(y)
        convolved_pdf_cdf /= convolved_pdf_cdf[-1]  # Normalize to get values between 0 and 1
        return convolved_pdf_cdf
    
    def _ppf(self,x):
        return self.inverse_cdf()(x)
    def inverse_cdf(self):
        if self.__inverse_cdf is None:
            x=np.linspace(0,np.pi,100000)
            self.__inverse_cdf = sp.interpolate.interp1d(self.cdf(x),x)
        return self.__inverse_cdf
############################################################