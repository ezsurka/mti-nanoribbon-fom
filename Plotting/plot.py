# Standard library imports
from pathlib import Path
import logging
from operator import itemgetter
from collections import defaultdict

# External package imports
import numpy as np
from numpy.ma import masked_array
from scipy import interpolate
import adaptive
from adaptive.utils import load
import matplotlib as mpl
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
#from pycolormap_2d import ColorMap2DTeuling2

# Internal imports
#import funcs
import pickle

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(message)s'
)

# add formatter to ch
ch.setFormatter(formatter)

if logger.hasHandlers():
    logger.handlers.clear()

# add ch to logger
logger.addHandler(ch)

redblackbule = LinearSegmentedColormap.from_list(
    "redbule", ["tab:blue", "black", "tab:red"]
)

gapcolor = 'tab:pink'
gaplinestyle = '-'#(0, (5, 5))


def set_size(rwidth, rhigh=0.6180339887498949):
    textwidth = 5.45776  # mdpi \textwidth in inch
    fig_width = textwidth * rwidth
    fig_height = fig_width * rhigh
    return fig_width, fig_height


def learner_from_file(fname, arg_picker_key=''):
    data = load(fname)

    if isinstance(data, tuple):
        domain = np.array(list(data[0].keys())).T
    else:
        domain = np.array(list(data.keys())).T

    if len(domain.shape) == 1:
        bounds = (domain.min(), domain.max())
    else:
        bounds = [(x.min(), x.max()) for x in domain]

    if isinstance(bounds, tuple):
        dim = 1
    else:
        dim = len(bounds)

    logger.info(f'Load {domain.shape[dim-1]} points form {fname}')

    if dim == 1:
        learner_type = adaptive.Learner1D
    elif dim == 2:
        learner_type = adaptive.Learner2D
    elif dim > 2:
        learner_type = adaptive.LearnerND

    if isinstance(data, tuple):
        arg_picker = itemgetter(arg_picker_key)
        learner_type = adaptive.make_datasaver(learner_type, arg_picker)

    learner = learner_type(function=None, bounds=bounds)

    learner._set_data(data)
    return learner


def get_learner_path(folder, prefix='data_learner_', **kwargs):
    folder = Path(folder)
    fname_params = []
    for k, v in kwargs.items():
        if isinstance(v, float):
            fname_params.append(f'{k}{v:3.4f}')
        else:
            fname_params.append(f'{k}{v}')
    fname = '_'.join(fname_params)
    fname = prefix + fname.replace('.', 'p') + '.pickle'
    return folder / fname


def learner_from_param(
        folder,
        prefix='data_learner_',
        arg_picker_key='',
        **kwargs
):
    fname = get_learner_path(folder, prefix, **kwargs)
    return learner_from_file(fname, arg_picker_key)


def learner_to_numpy(learner):
    points = np.array(list(learner.data.keys()))
    values = np.array(list(learner.data.values()), dtype=float)
    return points, values


def multi_color_line(xs, ys, c, ax=None, **kwargs):
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    lc = LineCollection(segments, **kwargs)
    lc.set_array(c)
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def resort_extra_data(extra_data):
    xs, _ys = zip(*sorted(extra_data.items()))
    ys = defaultdict(list)

    for y in _ys:
        for k, v in y.items():
            ys[k].append(v)

    for k in ys:
        ys[k] = ys[k]

    return xs, ys


def colored_line_plot(xs, ys, c, **kwargs):
    lcs = []
    for E, density in zip(ys.T, c.T):
        lcs.append(
            multi_color_line(xs, E, density, cmap=redblackbule, **kwargs)
        )
    return lcs


def map_values_to_syst(syst, values):
    syst_index = np.array([s.tag for s in syst.sites])
    syst_index_box = syst_index - np.amin(syst_index, 0)
    syst_box_shape = np.amax(syst_index, 0) - np.amin(syst_index, 0) + 1
    syst_box = np.full(syst_box_shape, np.nan)
    syst_box[tuple(syst_index_box.T)] = values
    return syst_box

def phase_diagram(
        folder,
        ax,
        file_prefix='data_learner_gap_',    
        gap_file_prefix='data_learner_gap_',
        mn_file_prefix='data_learner_mn_',
        n=500,
        **kwargs
):
    learner_mn = learner_from_param(
        folder,
        prefix=mn_file_prefix,
        **kwargs)

    learner_gap = learner_from_param(
        folder,
        prefix=gap_file_prefix,
        **kwargs)

    points_gap, values_gap = learner_to_numpy(learner_gap)
    points_mn, values_mn = learner_to_numpy(learner_mn)

    A_z = funcs.toy_A['A_z']
    T = kwargs['T']
    W = kwargs['W']
    delta = kwargs['delta']
    E_scale = 1
    logger.info(f'E_scale = {1e3*E_scale}')

    ip_gap = interpolate.LinearNDInterpolator(
        points_gap, values_gap, rescale=True
    )
    ip_mn = interpolate.NearestNDInterpolator(
        points_mn, values_mn, rescale=True
    )

    ((flux_min, flux_max), (mu_min, mu_max)) = learner_gap.bounds
    extent = [flux_min, flux_max, mu_min/E_scale, mu_max/E_scale]
    flux = np.linspace(flux_min, flux_max, num=n)
    mu = np.linspace(mu_min, mu_max, num=n)
    flux, mu = np.meshgrid(flux, mu)

    grid_gap = ip_gap(flux, mu)/delta
    grid_mn = ip_mn(flux, mu)
    mu = mu/E_scale

    grid_topo = masked_array(grid_gap, grid_mn == 1)

    norm = colors.LogNorm(vmin=1e-2, vmax=1)

    non_topo_im = ax.imshow(
        np.flip(grid_gap, 0),
        extent=extent,
        aspect="auto",
        cmap=mpl.cm.binary,
        norm=norm
    )

    topo_im = ax.imshow(
        np.flip(grid_topo, 0),
        extent=extent,
        aspect="auto",
        cmap=mpl.cm.Reds,
        norm=norm
    )

    cs = ax.contour(
        grid_mn,
        levels=[0],
        extent=extent,
        colors='tab:green',
        linewidths=1,
        linestyles='dashed'
    )

    return topo_im, non_topo_im, cs

def wave_function_plotter(
        folder,
        ax,
        file_prefix='data_learner_gap_',
        n=500,
        to_plot="left",
        **kwargs
):
    learner = learner_from_param(
        folder,
        prefix=file_prefix,
        **kwargs)
    
    points, values = learner_to_numpy(learner)
    
    data = list(learner.extra_data.values())
    values2plot_e = [data[i]["density_e_"+to_plot] for i in np.arange(0,len(data))]    
    values2plot_h = [data[i]["density_h_"+to_plot] for i in np.arange(0,len(data))]    
     
    ip_levels = interpolate.LinearNDInterpolator(
        points, values, rescale=True
    )    
    ip_e = interpolate.LinearNDInterpolator(
        points, values2plot_e, rescale=True
    )    
    ip_h = interpolate.LinearNDInterpolator(
        points, values2plot_h, rescale=True
    )       
        
    ((M_z_min, M_z_max), (mu_min, mu_max)) = learner.bounds
    extent = [M_z_min, M_z_max, mu_min, mu_max]
    M_z = np.linspace(M_z_min, M_z_max, num=n)
    mu = np.linspace(mu_min, mu_max, num=n)
    M_z, mu = np.meshgrid(M_z, mu)
    
    x,y = np.meshgrid(np.arange(0,500), np.arange(0,500))

    grid_e = ip_e(M_z, mu) 
    grid_h = ip_h(M_z, mu)     
    grid_levels = ip_levels(M_z, mu)

    cmap = ColorMap2DTeuling2()
    
    flat_e = np.nan_to_num(grid_e.flatten(),nan=-2)
    flat_h = np.nan_to_num(grid_h.flatten(),nan=-2)
    
    print("Max. electron density on "+to_plot+"="+str(max(flat_e)) )
    print("Max. hole density on "+to_plot+"="+str(max(flat_h)) )
    
    values2plot = [cmap(flat_e[i],1-flat_h[i]) for i in np.arange(0,len(flat_e))]
    
    values2plot = [np.array([71,0,71]) * (flat_e[i] < -1 or flat_h[i] < -1) + values2plot[i] for i in np.arange(0,len(flat_e))]
    
    values2plot = np.array(values2plot).reshape(len(M_z),len(mu),3)
    
    cs = ax.imshow(
        values2plot
    )
    
    cs_levels = ax.contour(
        x, y, grid_levels,
        levels=np.linspace(1e-2,int(max(values))+1,int(max(values))+2),
        extent=extent,
        colors='black'
    )
    
    M_z_ticks=[str(np.round(i,3)) for i in np.linspace(M_z_min, M_z_max,6)]
    mu_ticks =[str(np.round(i,3)) for i in np.linspace(mu_min, mu_max,6)]
    
    ax.set_xticks(np.arange(0,501,100));
    ax.set_xticklabels(M_z_ticks);
    ax.set_yticks(np.arange(0,501,100));
    ax.set_yticklabels(mu_ticks);

    ax.invert_yaxis();
    
    return M_z, mu, cs, cs_levels

def phase_plotter(
        folder,
        ax,
        file_prefix='data_learner_gap_',
        n=500,
        to_plot="E_gap",
        **kwargs
):
    learner = learner_from_param(
        folder,
        prefix=file_prefix,
        **kwargs)
    
    W = kwargs['W']
    delta = kwargs['delta']    
    
    points, values = learner_to_numpy(learner)
    data = list(learner.extra_data.values())
    
    try:
        values2plot = np.array([data[i][to_plot] for i in np.arange(0,len(data))])
    except:
        values2plot=0    
    
    if to_plot == "E_gap" or to_plot == "E_gap_2" or to_plot == "E_u" or to_plot == "E_l"  or to_plot == "E3_u" or to_plot == "E3_l":
        values = np.array([data[i]["E_gap"] for i in np.arange(0,len(data))])*1e3
        values2plot = np.array([data[i][to_plot] for i in np.arange(0,len(data))])*1e3
        
    cmin = 0
    cmax = max(values.flatten())
    cminmin = -1.01
    cmaxmax = max(values.flatten())*1.01    
    
    cmap=mpl.cm.rainbow
    
    if to_plot == "E_gap":
        cmin = 1e-12
        cminmin = 0
        cmax = 60
        cmaxmax = 1e4   
        values2plot[values2plot < 0] = -1e-12 #cut it off
        values2plot[values2plot < 0] = -1e-12        
    elif to_plot == "E_gap_2" or to_plot == "E_u" or to_plot == "E_l" or to_plot == "E3_u" or to_plot == "E3_l":
        cmin = 1e-12
        cminmin = 0
        cmax = 60
        cmaxmax = 1e4         
        values2plot[values2plot < 0] = -1e-12 #cut it off
        values2plot[values2plot < 0] = -1e-12          
        norm = colors.Normalize(vmin=0, vmax=60)
    elif to_plot == "modes":
        cmin = 0
        cmax = 8
        cmaxmax = max(8*1.01, cmaxmax)
    elif to_plot == "sigma_z_1" or to_plot == "sigma_z_2" :
        cmin = -1
        cmax = 1 
        cmaxmax = 1.01
    elif to_plot == "gap_u" or to_plot == "gap_l" or to_plot == "gap3_u" or to_plot == "gap3_l":
        cmap = mpl.cm.Reds
        values2plot = values2plot/delta
        values2plot[values2plot < 0] = -1e-12 #make it grey
        values2plot[values2plot < 0] = -1e-12
        norm = colors.Normalize(vmin=0, vmax=0.5)
    elif (to_plot == "fig_merit3_u" or to_plot == "fig_merit3_l") and kwargs['p_mode'] == 15:
        E_top_u = np.array([data[i]["E3_u"]   for i in np.arange(0,len(data))])/0.3
        gap_u   = np.array([data[i]["gap3_u"] for i in np.arange(0,len(data))])/delta
        E_top_l = np.array([data[i]["E3_l"]   for i in np.arange(0,len(data))])/0.3
        gap_l   = np.array([data[i]["gap3_l"] for i in np.arange(0,len(data))])/delta
        fig_merit_u = np.sqrt(E_top_u*gap_u)
        fig_merit_l = np.sqrt(E_top_l*gap_l)
        fig_merit_u[fig_merit_u > 1 ] = 0
        fig_merit_l[fig_merit_l > 1 ] = 0        
        cmap = mpl.cm.Blues        
        norm = colors.Normalize(vmin=0, vmax=.12 )
    elif to_plot == "fig_merit_u" or to_plot == "fig_merit_l":
        E_top_u = np.array([data[i]["E_u"]   for i in np.arange(0,len(data))])/0.3
        gap_u   = np.array([data[i]["gap_u"] for i in np.arange(0,len(data))])/delta
        E_top_l = np.array([data[i]["E_l"]   for i in np.arange(0,len(data))])/0.3
        gap_l   = np.array([data[i]["gap_l"] for i in np.arange(0,len(data))])/delta
        fig_merit_u = np.sqrt(E_top_u*gap_u)
        fig_merit_l = np.sqrt(E_top_l*gap_l)
        fig_merit_u[fig_merit_u > 1 ] = 0
        fig_merit_l[fig_merit_l > 1 ] = 0        
        cmap = mpl.cm.Blues        
        norm = colors.Normalize(vmin=0, vmax=.12 )
    else:
        values2plot = values
        
    if to_plot == "fig_merit_u" or to_plot == "fig_merit3_u":    
        values2plot = fig_merit_u
    elif to_plot == "fig_merit_l" or to_plot == "fig_merit3_l": 
        values2plot = fig_merit_l
        
    if kwargs['a_z'] < 0:
        if kwargs['p_mode'] == 1:
            shift = 1.12
        elif kwargs['p_mode'] == 15 and kwargs['a_z'] < -20:
            shift = 1.1525
        elif kwargs['p_mode'] == 15 and kwargs['a_z'] == -20:
            shift = 1.0425
        else:
            shift = 1
    else:
        shift = 1
    
#    if kwargs['sym'] == True:
#        hybridization_gaps = hybridization_gaps_symTrue
#    else:
#        hybridization_gaps = hybridization_gaps_symFalse
#    pos = np.where(hybridization_gaps[0] == kwargs['p_mode'])[0][0]
#    ax.plot(hybridization_gaps[2][pos],hybridization_gaps[1][pos]/shift, 'k-')
#    ax.plot(-hybridization_gaps[2][pos],hybridization_gaps[1][pos]/shift, 'k-')
    
    ((M_z_min, M_z_max), (mu_min, mu_max)) = learner.bounds
    extent = [-M_z_max, M_z_max, mu_min, mu_max]        
    
    ax.set_xlim(-M_z_max, M_z_max)
    ax.set_ylim(mu_min, mu_max)    
    
    M_z = np.linspace(M_z_min, M_z_max, num=n)
    mu = np.linspace(mu_min, mu_max, num=n)
    M_z, mu = np.meshgrid(M_z, mu)
    
    M_z_plot  = np.concatenate((np.flip(-M_z,axis=1), M_z),axis=1)
    mu_plot  = np.concatenate((mu, mu),axis=1)
    
    ip = interpolate.LinearNDInterpolator(
        points, values2plot, rescale=True
    )       

    grid = ip(M_z, mu)
    grid = np.concatenate((np.flip(grid,axis=1), grid),axis=1)
    
    if (to_plot == "gap_u"  or to_plot == "gap_l"  or to_plot == "fig_merit_u"  or to_plot == "fig_merit_l"  or to_plot == "E_u"  or to_plot == "E_l" or
        to_plot == "gap3_u" or to_plot == "gap3_l" or to_plot == "fig_merit3_u" or to_plot == "fig_merit3_l" or to_plot == "E3_u" or to_plot == "E3_l") and delta > 0:

        if to_plot[-3] == '3':
            ip_mn = interpolate.NearestNDInterpolator(
                points,  np.array([data[i]["mn3"+to_plot[-2:]] for i in np.arange(0,len(data))]), rescale=True
            )
        else:
            ip_mn = interpolate.NearestNDInterpolator(
                points,  np.array([data[i]["mn"+to_plot[-2:]] for i in np.arange(0,len(data))]), rescale=True
            )
        
        grid_mn = ip_mn(M_z, mu)
        grid_mn = np.concatenate((np.flip(grid_mn,axis=1), grid_mn),axis=1)    
        
        grid = ip(M_z, mu)
        grid = np.concatenate((np.flip(grid,axis=1), grid),axis=1)        
        grid_topo = masked_array(grid, grid_mn != -1)

        non_topo_im = ax.imshow(
            np.flip(grid, 0),
            extent=extent,
            aspect="auto",
            cmap=mpl.cm.binary,
            norm=colors.Normalize(vmin=-100, vmax=100),
        )

        topo_im = ax.imshow(
            np.flip(grid_topo, 0),
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=norm
        )
        
        ax.set_ylim(20/shift, 100/shift);
        ax.set_yticks(np.arange(20,101,10)/shift);
        ax.set_yticklabels(np.arange(20,101,10));
        
        return topo_im
    else:
        cs = ax.contourf(
            M_z_plot, mu_plot, grid,
            levels=[cminmin]+list(np.linspace(cmin,cmax,100))+[cmaxmax],
            extent=extent,
            cmap=cmap,
            vmin=cmin,vmax=cmax,
        )

        ax.set_ylim(20/shift, 100/shift);
        ax.set_yticks(np.arange(20,101,10)/shift);
        ax.set_yticklabels(np.arange(20,101,10));
        
    if to_plot == "sigma_z_1" or to_plot == "sigma_z_2":
        ip = interpolate.LinearNDInterpolator(
            points, values, rescale=True
        )         
        cs_levels = ax.contour(
            M_z, mu, ip(M_z, mu),
            levels=np.linspace(1e-2,int(max(values))+1,int(max(values))+2),
            extent=extent,
            colors='black'
        )        
        return cs, cs_levels

    return cs

def phase_plotter_2D(
        folder,
        ax,
        file_prefix='data_learner_gap_',
        n=500,
        to_plot="E_gap",
        **kwargs
):
    
    data = []
    
    params = kwargs.copy()
    for T in kwargs['T']:
        params['T'] = T
        learner = learner_from_param(
            folder,
            prefix=file_prefix,
            **params)
        data.append(learner.extra_data)
    
    W = kwargs['W']
    delta = kwargs['delta']    
    
    values2plot = [[data[i][j][to_plot] for j in data[i].keys()] for i in range(len(data)) ]

    return data, [data[i].keys() for i in range(len(data))] ,values2plot
    
    if to_plot == "E_gap" or to_plot == "E_gap_2" or to_plot == "E_u" or to_plot == "E_l":
        values = np.array([data[i]["E_gap"] for i in np.arange(0,len(data))])*1e3
        values2plot = np.array([data[i][to_plot] for i in np.arange(0,len(data))])*1e3
        
    cmin = 0
    cmax = max(values.flatten())
    cminmin = -1.01
    cmaxmax = max(values.flatten())*1.01    
    
    cmap=mpl.cm.rainbow
    
    if to_plot == "E_gap":
        cmin = 1e-12
        cminmin = 0
        cmax = 60
        cmaxmax = 1e4   
        values2plot[values2plot < 0] = -1e-12 #cut it off
        values2plot[values2plot < 0] = -1e-12        
    elif to_plot == "E_gap_2" or to_plot == "E_u" or to_plot == "E_l":
        cmin = 1e-12
        cminmin = 0
        cmax = 60
        cmaxmax = 1e4         
        values2plot[values2plot < 0] = -1e-12 #cut it off
        values2plot[values2plot < 0] = -1e-12          
        norm = colors.Normalize(vmin=0, vmax=60)
    elif to_plot == "modes":
        cmin = 0
        cmax = 8
        cmaxmax = max(8*1.01, cmaxmax)
    elif to_plot == "sigma_z_1" or to_plot == "sigma_z_2" :
        cmin = -1
        cmax = 1 
        cmaxmax = 1.01
    elif to_plot == "gap_u" or to_plot == "gap_l":
        cmap = mpl.cm.Reds
        values2plot = values2plot/delta
        values2plot[values2plot < 0] = -1e-12 #make it grey
        values2plot[values2plot < 0] = -1e-12
        norm = colors.Normalize(vmin=0, vmax=0.5)    
    elif to_plot == "fig_merit_u" or "fig_merit_l":
        E_top_u = np.array([data[i]["E_u"] for i in np.arange(0,len(data))])/0.3
        gap_u   = np.array([data[i]["gap_u"] for i in np.arange(0,len(data))])/delta
        E_top_l = np.array([data[i]["E_l"] for i in np.arange(0,len(data))])/0.3
        gap_l  = np.array([data[i]["gap_l"] for i in np.arange(0,len(data))])/delta
        fig_merit_u = np.sqrt(E_top_u*gap_u)
        fig_merit_l = np.sqrt(E_top_l*gap_l)
        fig_merit_u[fig_merit_u > 1 ] = 0
        fig_merit_l[fig_merit_l > 1 ] = 0        
        cmap = mpl.cm.Blues        
        norm = colors.Normalize(vmin=0, vmax=max(max(fig_merit_u),max(fig_merit_l)) )                
    else:
        values2plot = values
        
    if to_plot == "fig_merit_u":    
        values2plot = fig_merit_u
    elif to_plot == "fig_merit_l": 
        values2plot = fig_merit_l
        
    if kwargs['a_z'] == -20:
        shift = 1.08
    elif kwargs['a_z'] == -10:
        shift = 1.187
    else:
        shift = 1 
    
#    ax.set_yticks(np.arange(20,110,10)/shift,np.arange(20,110,10));
    
    if kwargs['sym'] == True:
        hybridization_gaps = hybridization_gaps_symTrue
    else:
        hybridization_gaps = hybridization_gaps_symFalse
    pos = np.where(hybridization_gaps[0] == kwargs['p_mode'])[0][0]
    ax.plot(hybridization_gaps[2][pos],hybridization_gaps[1][pos]/shift, 'k-')
    ax.plot(-hybridization_gaps[2][pos],hybridization_gaps[1][pos]/shift, 'k-')
    
    ((M_z_min, M_z_max), (mu_min, mu_max)) = learner.bounds
    extent = [-M_z_max, M_z_max, mu_min, mu_max]        
    
    ax.set_xlim(-M_z_max, M_z_max)
    ax.set_ylim(mu_min, mu_max)    
    
    M_z = np.linspace(M_z_min, M_z_max, num=n)
    mu = np.linspace(mu_min, mu_max, num=n)
    M_z, mu = np.meshgrid(M_z, mu)
    
    M_z_plot  = np.concatenate((np.flip(-M_z,axis=1), M_z),axis=1)
    mu_plot  = np.concatenate((mu, mu),axis=1)
    
    ip = interpolate.LinearNDInterpolator(
        points, values2plot, rescale=True
    )       

    grid = ip(M_z, mu)
    grid = np.concatenate((np.flip(grid,axis=1), grid),axis=1)
    
    if (to_plot == "gap_u" or to_plot == "gap_l" or to_plot == "fig_merit_u" or  to_plot == "fig_merit_l" or to_plot == "E_u" or to_plot == "E_l") and delta > 0:

        ip_mn = interpolate.NearestNDInterpolator(
            points,  np.array([data[i]["mn"+to_plot[-2:]] for i in np.arange(0,len(data))]), rescale=True
        )
        
        grid_mn = ip_mn(M_z, mu)
        grid_mn = np.concatenate((np.flip(grid_mn,axis=1), grid_mn),axis=1)    
        
        grid = ip(M_z, mu)
        grid = np.concatenate((np.flip(grid,axis=1), grid),axis=1)        
        grid_topo = masked_array(grid, grid_mn != -1)

        non_topo_im = ax.imshow(
            np.flip(grid, 0),
            extent=extent,
            aspect="auto",
            cmap=mpl.cm.binary,
            norm=colors.Normalize(vmin=-100, vmax=100),
        )

        topo_im = ax.imshow(
            np.flip(grid_topo, 0),
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=norm
        )
        
        return topo_im
    else:
        cs = ax.contourf(
            M_z_plot, mu_plot, grid,
            levels=[cminmin]+list(np.linspace(cmin,cmax,100))+[cmaxmax],
            extent=extent,
            cmap=cmap,
            vmin=cmin,vmax=cmax,
        )
    if to_plot == "sigma_z_1" or to_plot == "sigma_z_2":
        ip = interpolate.LinearNDInterpolator(
            points, values, rescale=True
        )         
        cs_levels = ax.contour(
            M_z, mu, ip(M_z, mu),
            levels=np.linspace(1e-2,int(max(values))+1,int(max(values))+2),
            extent=extent,
            colors='black'
        )        
        return cs, cs_levels

    return cs


def spectrum(
        folder,
        ax,
        delta_zero=False,
        gap_line=True,
        E_scale=1,
        **kwargs
):
    a = kwargs['a']
    delta = kwargs['delta']
    if delta_zero != 'only':
        learner = learner_from_param(folder, arg_picker_key='Es', **kwargs)
        xs, ys = resort_extra_data(learner.extra_data)
        ks = np.array(xs)
        Es = np.array(ys['Es'])
        rhos_p = np.array(ys['rhos_p'])[:, 0, :]
        rhos_h = np.array(ys['rhos_h'])
        densitys_ph = rhos_p - rhos_h

        norm = colors.Normalize(-1, 1)
        if delta != 0:
            E_scale = delta

        lcs = colored_line_plot(
            10*ks/a, Es/E_scale, densitys_ph, norm=norm, ax=ax, rasterized=True
        )

        if gap_line:
            logger.info(f'gap = {Es[Es > 0].min()/E_scale}')
            ax.axhline(
                y=Es[Es > 0].min()/E_scale,
                linestyle=(0, (5, 10)),
                color=gapcolor,
                linewidth=0.25
            )
            ax.axhline(
                y=-Es[Es > 0].min()/E_scale,
                linestyle=(0, (5, 10)),
                color=gapcolor,
                linewidth=0.25
            )

    if delta_zero:
        kwargs['delta'] = 0
        learner = learner_from_param(folder, arg_picker_key='Es', **kwargs)
        ks, Es = zip(*sorted(learner.data.items()))
        ks = np.array(ks)
        Es = np.stack(Es)
        if delta_zero == 'only':
            line = ax.plot(
                10*ks/a, Es/E_scale, '.', markersize=2, markeredgewidth=0,
                color='black', zorder=10, rasterized=True
            )
            return line
        else:
            line = ax.plot(
                10*ks/a, Es/E_scale, '-',
                color='black', linewidth=0.1, zorder=0, rasterized=True
            )
        return lcs, line
    else:
        return lcs


def finite_spectrum(folder_spectrum, folder_gap, ax, **kwargs):
    learner = learner_from_param(
        folder_spectrum, arg_picker_key='Es', **kwargs
    )
    xs, ys = resort_extra_data(learner.extra_data)
    fluxs = np.array(xs)
    Es = np.array(ys['Es_mean'])
    rhos_p = np.array(ys['rhos_p_mean'])
    rhos_h = np.array(ys['rhos_h_mean'])
    densitys_ph = rhos_p - rhos_h

    delta = kwargs['delta']

    norm = colors.Normalize(-1, 1)
    lcs = colored_line_plot(
        fluxs, Es/delta, densitys_ph, norm=norm,
        ax=ax, linewidth=0.3, rasterized=True
    )

    if folder_gap:
        learner_gap = learner_from_param(
            folder_gap,
            prefix='data_learner_gap_',
            arg_picker_key='gap',
            a=kwargs['a'],
            T=kwargs['T'],
            W=kwargs['W'],
            delta=kwargs['delta'],
            m_z=kwargs['m_z'],
            mu=kwargs['mu_ti'],
        )
        fluxs, gap = zip(*sorted(learner_gap.data.items()))
        fluxs = np.array(fluxs)
        gap = np.array(gap)

        ax.plot(
            fluxs, gap/delta, linestyle=gaplinestyle,
            color=gapcolor, linewidth=0.5
        )
        ax.plot(
            fluxs, -gap/delta, linestyle=gaplinestyle,
            color=gapcolor, linewidth=0.5
        )

    return lcs


def zbp2d(
        folder_c,
        folder_gap,
        ax,
        prefix='data_learner_c_',
        norm=None,
        cmap=None,
        n=1000,
        **kwargs
):
    learner_c = learner_from_param(
        folder_c, prefix=prefix, **kwargs
    )
    points_c, values_c = learner_to_numpy(learner_c)

    B, E, c = learner_c.interpolated_on_grid(n=n)

    delta = kwargs['delta']

    pc = ax.pcolormesh(
        B, E/delta, np.rot90(c),
        norm=norm,
        rasterized=True,
        cmap=cmap
    )

    learner_gap = learner_from_param(
        folder_gap,
        prefix='data_learner_gap_',
        arg_picker_key='gap',
        a=kwargs['a'],
        T=kwargs['T'],
        W=kwargs['W'],
        delta=kwargs['delta'],
        m_z=kwargs['m_z'],
        mu=kwargs['mu_ti'],
    )

    fluxs, gap = zip(*sorted(learner_gap.data.items()))

    fluxs = np.array(fluxs)
    gap = np.array(gap)

    ax.plot(
        fluxs, gap/delta, linestyle=gaplinestyle,
        color=gapcolor, linewidth=0.5
    )
    ax.plot(
        fluxs, -gap/delta, linestyle=gaplinestyle,
        color=gapcolor, linewidth=0.5
    )

    return pc


def zbp_line_cut(
        folder_c,
        folder_gap,
        ax,
        prefix='data_learner_c_',
        n=500,
        flux=0.5,
        **kwargs
):
    learner_c = learner_from_param(
        folder_c, prefix=prefix, **kwargs
    )
    points_c, values_c = learner_to_numpy(learner_c)
    delta = kwargs['delta']

    ((flux_min, flux_max), (E_min, E_max)) = learner_c.bounds
    points_c, values_c = learner_to_numpy(learner_c)
    E = np.linspace(E_min, E_max, num=n)
    flux = np.repeat(flux, repeats=n)
    interp = interpolate.LinearNDInterpolator(points_c, values_c, rescale=True)
    Smag_imp = kwargs['Smag_imp']
    A_z = funcs.toy_A['A_z']
    T = kwargs['T']
    W = kwargs['W']
    delta = kwargs['delta']
    E_scale = np.pi*A_z/(T+W)

    if Smag_imp == 0.0:
        label = (
            r"$S_\mathrm{dis}$ "
            f"= {Smag_imp/E_scale:2.0f}"
        )
    else:
        label = (
            r"$S_\mathrm{dis}$ "
            f"= {Smag_imp/E_scale:2.1f}"
            r"$\times 2\pi A/P$"
        )

    _ = ax.plot(
        E/delta, interp(flux, E),
        label=label
    )

    if folder_gap:
        learner_gap = learner_from_param(
            folder_gap,
            prefix='data_learner_gap_',
            arg_picker_key='gap',
            a=kwargs['a'],
            T=kwargs['T'],
            W=kwargs['W'],
            delta=kwargs['delta'],
            m_z=kwargs['m_z'],
            mu=kwargs['mu_ti'],
        )
        points_gap, values_gap = learner_to_numpy(learner_gap)
        ip_gap = interpolate.interp1d(
            points_gap, values_gap
        )
        gap = ip_gap([flux[0]])[0]/delta
        logger.info(f'gap = {gap}')
        _ = ax.axvline(x=gap, linestyle='--', color=gapcolor)
        _ = ax.axvline(x=-gap, linestyle='--', color=gapcolor)


def zbp1d(
        folder_c,
        folder_gap,
        ax,
        prefix='data_learner_c_1d_',
        **kwargs
):
    learner_c = learner_from_param(
        folder_c, prefix=prefix, **kwargs)

    data = learner_c.to_numpy()

    delta = kwargs['delta']
    Smag_imp = kwargs['Smag_imp']
    A_z = funcs.toy_A['A_z']
    T = kwargs['T']
    W = kwargs['W']
    delta = kwargs['delta']
    E_scale = np.pi*A_z/(T+W)

    if Smag_imp == 0.0:
        label = (
            r"$S_\mathrm{dis}$ "
            f"= {Smag_imp/E_scale:2.0f}"
        )
    else:
        label = (
            r"$S_\mathrm{dis}$ "
            f"= {Smag_imp/E_scale:2.1f}"
            r"$\times 2\pi A/P$"
        )

    line = ax.plot(
        data[:, 0]/delta, data[:, 1],
        label=label
    )

    if folder_gap:
        learner_gap = learner_from_param(
            folder_gap,
            prefix='data_learner_gap_',
            arg_picker_key='gap',
            a=kwargs['a'],
            T=kwargs['T'],
            W=kwargs['W'],
            delta=kwargs['delta'],
            m_z=kwargs['m_z'],
            mu=kwargs['mu_ti'],
        )
        points_gap, values_gap = learner_to_numpy(learner_gap)
        ip_gap = interpolate.interp1d(
            points_gap, values_gap
        )
        flux = kwargs['flux']
        gap = ip_gap(flux)/delta
        logger.info(f'gap = {gap}')
        _ = ax.axvline(x=gap, linestyle='--', color=gapcolor)
        _ = ax.axvline(x=-gap, linestyle='--', color=gapcolor)

    return line


def gap_1d_mu(
        folder,
        ax,
        prefix='data_learner_gap_1d_',
        **kwargs
):
    learner = learner_from_param(
        folder=folder,
        prefix=prefix,
        **kwargs
    )

    data = learner.to_numpy()

    A_z = funcs.toy_A['A_z']
    T = kwargs['T']
    W = kwargs['W']
    delta = kwargs['delta']
    E_scale = np.pi*A_z/(T+W)

    pos_discontinuity = np.where(np.abs(np.diff(data[:, 1]/delta)) >= 0.5)[0]
    data[pos_discontinuity, 0] = np.nan
    data[pos_discontinuity, 1] = np.nan

    ax.plot(
        data[:, 0]/E_scale, data[:, 1]/delta, label=r'$\min\{E(k=0)\}$'
    )

    def plot_line(i, label):
        gap = data[:, i]
        ax.plot(
            data[:, 0]/E_scale, gap/delta, label=label
        )

    plot_line(2, 'FP 1')
    plot_line(4, 'FP 2 \\& 3')
    plot_line(6, 'FP 4 \\& 5')


def densety(
        ax,
        density_map,
        flux, mu,
        Smag_imp,
        sum_ind,
        E_scale
):
    density = density_map[flux, mu, Smag_imp]
    density = density.reshape(8, 1001, 11, 11)
    density = density[4-sum_ind:4+sum_ind]
    density = np.sum(density, axis=(2, 3))
    debsety_sums = np.sum(density, axis=1)

    for i, debsety_sum in enumerate(debsety_sums):
        density[i] = density[i]/debsety_sum

    density = np.sum(density, axis=0)
    num_sides = density.shape[0]
    x = np.linspace(0, 1, num=num_sides)

    if Smag_imp == 0.0:
        label = (
            r"$S_\mathrm{dis}$ "
            f"= {Smag_imp/E_scale:2.0f}"
        )
    else:
        label = (
            r"$S_\mathrm{dis}$ "
            f"= {Smag_imp/E_scale:2.1f}"
            r"$\times 2\pi A/P$"
        )

    line = ax.plot(
        x, num_sides*density,
        label=label
    )
    return line

def combo_fname(directory, params, suffix=""):
    fname_params = []
    for k, v in params.items():
        if isinstance(v, float):
            fname_params.append(f'{k}{v:3.4f}')
        else:
            fname_params.append(f'{k}{v}')

    fname = '_'.join(fname_params)
    fname = "./figures/" + directory["file_prefix"]+suffix+fname.replace('.', 'p')
    
    return fname