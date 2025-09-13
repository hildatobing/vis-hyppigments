# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:07:56 2017

Note:
05/02/2025  This is a version of the master distance_functions file created
            for the hyperspectral pigments interface.

@author: Hilda Deborah
"""

import numpy as np
import numpy.linalg as LA


def interpolate(
        target_spectrum, target_wvl, ref_spectra, ref_wvl):
    """
    Target spectrum is always interpolated to the reference wavelength. But wa-
    velength subset is always considered both directions.
    """
    if min(target_wvl) < min(ref_wvl):
        min_idx = np.argmin(np.abs(target_wvl - min(ref_wvl)))
        target_spectrum = target_spectrum[min_idx:]
        target_wvl = target_wvl[min_idx:]
    else:
        min_idx = np.argmin(np.abs(ref_wvl - min(target_wvl)))
        ref_spectra = ref_spectra[:, min_idx:]
        ref_wvl = ref_wvl[min_idx:]
    
    if max(target_wvl) > max(ref_wvl):
        max_idx = np.argmin(np.abs(target_wvl - max(ref_wvl)))
        target_spectrum = target_spectrum[:max_idx+1]
        target_wvl = target_wvl[:max_idx+1]
    else:
        max_idx = np.argmin(np.abs(ref_wvl - max(target_wvl)))
        ref_spectra = ref_spectra[:, :max_idx+1]
        ref_wvl = ref_wvl[:max_idx+1]

    interp_target = np.interp(ref_wvl, target_wvl, target_spectrum)
    return interp_target, ref_spectra, ref_wvl


def get_distance_dict():
    """
    Return: List of all distance functions made available for the interface.
    """
    return {
        'klpd':'KLPD - Kullback-Leibler pseudo-divergence',
        'klpde':'eKLPD - Energy component of KLPD',
        'klpds':'sKLPD - Shape component of KLPD',
        'rms':'RMSE - Root Mean Square Error', 
        'manhattan':'Manhattan distance',
        'euclidean':'Euclidean distance',
        'sqeuc':'Squared Euclidean distance',
        'chebyshev':'Chebyshev distance',
        'sam':'SAM - Spectral Angle Mapper', 
        'sid':'SID - Jeffrey divergence / Spectral Information Divergence',
        'gfc':'GFC - Goodness-of-Fit Coefficient',
        'cos':'Cosine distance',
        'scorr':'Spectral Correlation',
        'jensen':'JS divergence - Jensen-Shannon',
        'kl':'KL divergence - Kullback-Leibler',
        'empjeff':'Empirical Jeffrey divergence (histogram)',
        'canberra':'L1 - Canberra distance',
        'cumL1':'Cumulative L1, or energy differences',
        'sqchord':'Squared Chord distance'}


def replicate_ref(ref, r, c, d):
    """
    Replicate reference spectrum matrix to the size of target spectra so that
    distance computation can be done in matrix operation instead of iteration.
    If ref is already the shape of r, c, d, return ref.

    Parameters:
    - `ref`: reference spectrum, of dimension 1x1x(# of wavelengths).
    - `r`: number of rows in the target matrix.
    - `c`: number of columns in the target matrix.
    - `d`: number of wavelengths in the target matrix.

    Returns: replicated reference spectrum, of equal dimension as the
    target spectra matrix.
    """
    if len(ref.shape) == 1:
        ref = ref.reshape(1, 1, ref.shape[0])
    r1, c1 = ref.shape[:2]
    if r1 == r and c1 == c:
        return ref
    else:
        return np.tile(ref, r * c).reshape(r, c, d)
    

def get_distance_values(ref, B, fun, resolution=1., alpha=1.):
    """
    Interprets the distance function stringname and calls its
    corresponding method.

    Parameters:
    - `ref`: reference spectrum to calculate distance from, of
      dimension 1x1x(# of wavelength)
    - `B`: target spectra, of 3d image dimension, i.e.
      (# of rows)x(# of cols)x(# of wavelength)
    - `fun`: distance function stringname. In case of manifold
      distance, this string specifies the embedded distance function
      inside the manifold distance, which measures distance between
      two spectra.

    Return: Distance values between reference spectrum and all target
    spectra for the given distance function.
    """
    fun = fun.lower()

    ndim = len(B.shape)
    row, col, wvl = 0, 0, 0
    if ndim == 3:
        row, col, wvl = B.shape
    elif ndim == 2:
        nentries, nwvls = B.shape
        B = B.reshape(1, nentries, nwvls)
        row, col, wvl = B.shape
        ref = ref.reshape(1, 1, nwvls)

    A = replicate_ref(ref, row, col, wvl)
    distance_values = np.zeros((row, col), np.double)

    # Root mean square error
    if fun == 'rms':
        distance_values = rms_error(A, B)
    # Manifold
    elif fun == 'gfc':
        distance_values = gfc_sam_cos(A, B, param='gfc')
        distance_values[distance_values <= 1e-6] = 0
    # Angular
    elif fun == 'sam':
        distance_values = gfc_sam_cos(A, B, param='sam')
        distance_values[distance_values <= 1e-6] = 0
    elif fun == 'cos':
        distance_values = gfc_sam_cos(A, B, param='cos')
        distance_values[distance_values <= 1e-6] = 0
    # Correlation
    elif fun == 'scorr':
        distance_values = scm(A, B, corr=True)
    # Minkowski
    elif fun == 'manhattan':
        distance_values = minkowski(A, B, 1)
    elif fun == 'euclidean':
        distance_values = minkowski(A, B, 2)
    elif fun == 'chebyshev':
        distance_values = chebyshev(A, B)
    # Divergences
    elif fun == 'sid':
        distance_values = sid(A, B, resolution=resolution)
        distance_values[distance_values <= 1e-6] = 0
    elif fun == 'empjeff':
        distance_values = empirical_jeffrey(A, B, ishist=True)
    elif fun == 'jensen':
        distance_values = empirical_jeffrey(A, B, jensen=True)
    # Pseudo-divergences
    elif fun == 'kl':
        distance_values = KL(normalize_spectra(A), normalize_spectra(B))
        
    elif fun == 'klpd':
        distance_values = pseudodiv_KL(A, B, resolution=resolution, mode=3)
    elif fun == 'klpds':
        distance_values = pseudodiv_KL(A, B, resolution=resolution, mode=1)
    elif fun == 'klpde':
        distance_values = pseudodiv_KL(A, B, resolution=resolution, mode=2)
    # Energy differences
    elif fun == 'cumL1':
        distance_values = cumulative_L1(A, B)

    # L1
    elif fun == 'canberra':
        distance_values = canberra(A, B)

    # L2
    elif fun == 'sqeuc':
        distance_values = squared_L2(A, B)

    elif fun == 'sqchord':
        distance_values = squared_chord(A, B)

    else:
        print(fun, 'distance function not found')
        return

    return np.nan_to_num(distance_values)


def squared_L2(A, B):
    return np.power(minkowski(A, B, 2), 2.)


def cumulative_L1(A, B):
    """
    Energy difference, i.e. Manhattan distance of cumulative spectrum (ECS).

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: distance matrix, of dimension rowxcol
    """
    A = A.cumsum(axis=2)
    B = B.cumsum(axis=2)
    return minkowski(A, B, 1)


def cumulative_L2(A, B):
    """
    Euclidean distance of cumulative spectrum (ECS).

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: distance matrix, of dimension rowxcol
    """
    A = A.cumsum(axis=2)
    B = B.cumsum(axis=2)
    return minkowski(A, B, 2)


def squared_chord(A, B):
    '''
    Squared chord, not applicable for feature spaces with negative
    values.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: distance matrix, of dimension rowxcol
    '''
    return np.sum(np.power(np.sqrt(A) - np.sqrt(B), 2.), axis=2)


def canberra(A, B):
    '''
    Canberra metric.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: distance matrix, of dimension rowxcol
    '''
    return np.sum(np.divide(np.abs(A - B), (np.abs(A) + np.abs(B))), axis=2)


def empirical_jeffrey(A, B, jensen=False, ishist=False):
    '''
    Empirical jeffrey divergence.
    Note: 
    28/10/2014  log10 changed into log (ln). In literatures ln is used, however
                the difference between log10 and log is only in its dynamic
                range (dnr of ln is bigger); saturation point remains the same.
    20/04/2016  Name changed into empirical jeffreys and input spectra are
                always normalized before use. Initially this is the Jeffrey's
                divergence used in JSTARS.

    Arguments:
        A(np.array):
            Reference matrix, of dimension rowxcolxwavelength.

        B(np.array):
            Target matrix, of dimension rowxcolxwavelength.

        ishist(bool):
            Whether input shall be considered as PDF or histogram. In case of
            PDF (ishist==False), they have to be normalized.

    Returns: distance matrix, of dimension rowxcol
    '''
    if not ishist:
        A = normalize_spectra(A)
        B = normalize_spectra(B)
    m = np.divide((A + B), 2)
    a = np.multiply(np.log(np.divide(A, m)), A)
    b = np.multiply(np.log(np.divide(B, m)), B)
    jef = np.sum((np.nan_to_num(a) + np.nan_to_num(b)), axis=2)
    if jensen:
        return np.multiply(jef, 2)
    else:
        return jef


def sid(A, B, resolution=1.):
    '''
    Spectral information divergence, essentially identical to Jeffrey's
    divergence.

    Arguments:
        A(np.array):
            Reference matrix, of dimension rowxcolxwavelength.

        B(np.array):
            Target matrix, of dimension rowxcolxwavelength.

        ishist(bool):
            Whether input shall be considered as PDF or histogram. In case of
            PDF (ishist==False), they have to be normalized.

    Returns: distance matrix, of dimension rowxcol
    '''
    p_A = normalize_spectra(A, resolution=resolution)
    p_B = normalize_spectra(B, resolution=resolution)
    p_A[p_A <= 0.] = 1e-6
    p_B[p_B <= 0.] = 1e-6
    return KL(p_A, p_B, resolution=resolution) + KL(
            p_B, p_A, resolution=resolution)
    

def pseudodiv_KL(A, B, resolution=1., mode=0):
    """
    Kullback-Leibler pseudo-divergence for spectral data, integration method is
    assumed to be trapezoidal.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.
    - `mode`: whether to return both components (default, 0), shape(1), 
        energy(2), or total (summation, 3)

    Return: distance matrix, of dimension rowxcol
    """    
    if len(A.shape) == 1:
        # This handles pairwise distance with this function as metric callable
        A = A[np.newaxis, np.newaxis, :]
        B = B[np.newaxis, np.newaxis, :]
        # If mode is not setup, by default total klpd is given
        if mode == 0:
            mode = 3
        
    kA, n_A = normalize_spectra(A, get_w=True, resolution=resolution)
    kB, n_B = normalize_spectra(B, get_w=True, resolution=resolution)
    shape = (kA * KL(n_A, n_B, resolution=resolution)) + (
            kB * KL(n_B, n_A, resolution=resolution))
    energy = (kA - kB) * (np.log(kA) - np.log(kB))

    if mode == 0:
        return np.concatenate((shape[:, :, None], energy[:, :, None]), axis=2)
    elif mode == 1:
        return shape
    elif mode == 2:
        return energy
    else:
        return shape + energy


def KL(A, B, resolution=1.):
    """
    Kullback-Leibler, the original divergence. Input is assumed to be
    normalized to one.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: divergence matrix, of dimension rowxcol
    """
    scale = 1e6
    part_A = np.nan_to_num(np.log(np.multiply(A, scale)) - np.log(scale))
    part_B = np.nan_to_num(np.log(np.multiply(B, scale)) - np.log(scale))
    div_KL = np.multiply(A, (part_A - part_B))
    return np.trapz(div_KL, dx=resolution, axis=2)


def normalize_spectra(A, get_w=False, resolution=1.):
    '''
    Normalize each spectrum to the sum of values at each of its wavelengths. If
    integration is True, the normalizing factor is integration instead,
    trapezoidal rule is assumed.

    Arguments
        A(np.array):
            Input matrix, of dimension rowxcolxwavelength.

        get_w(bool):
            Whether to return the normalizing factor.

        integration(bool):
            Whether to use integration as normalizing factor.

    Returns: `A` normalized into probability matrix.
    '''
    A[A <= 0.] = 1e-9  # Handling for zero values

    r, c, b = np.shape(A)
    norm_factor = np.trapz(A, dx=resolution, axis=2)
    norm_factor = norm_factor.reshape(r, c, 1)
    norm_factor = np.tile(norm_factor, (1, 1, b))
    if get_w:
        return norm_factor[:, :, 0], np.divide(A, norm_factor)
    else:
        return np.divide(A, norm_factor)


def chebyshev(A, B):
    """
    One of the variety of Minkowski, with p equals infinity.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: distance matrix, of dimension rowxcol
    """
    return np.max(np.absolute(A - B), axis=2)


def minkowski(A, B, p):
    """
    Minkowski distance functions.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.
    - `p`: Minkowski parameter.

    Return: distance matrix, of dimension rowxcol
    """
    return np.power(np.sum(np.power(np.absolute(A - B), p), axis=2), (1. / p))


def rms_error(A, B):
    """
    RMSE distance functions.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: distance matrix, of dimension rowxcol
    """
    return np.sqrt(np.sum(np.power(A - B, 2.), axis=2) * (1. / len(A[0])))


def gfc_sam_cos(A, B, param=''):
    """
    Goodness-of-Fit Coefficient and Spectral Angle Mapper. Compute similarity 
    between spectra using cosine function. SAM = arccos(GFC). Value of SAM 
    similarity varies between 0 and 1, closer to 0 means higher similarity.

    However pay attention that the original formula of GFC takes the absolute 
    values of the sum of element multiplications. Here it's not implemented in 
    such way since we are assuming spectral reflectance data which is always 
    positive.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: distance matrix, of dimension rowxcol
    """
    A[A <= 0.] = 1e-6
    B[B <= 0.] = 1e-6
    
    divident = np.sum(np.multiply(A, B), 2)
    divisorA = np.apply_along_axis(LA.norm, 2, A)
    divisorB = np.apply_along_axis(LA.norm, 2, B)
    divisor = np.multiply(divisorA, divisorB)
    cosine = np.divide(divident, divisor)
    cosine[divident == 0] = 1
    cosine[cosine == np.nan] = 1
    cosine[cosine > 1.] = 1
    
    if param == 'sam':
        return np.arccos(cosine)
    elif param == 'gfc':
        return 1 - cosine
    elif param == 'cos':
        return 1 - cosine


def scm(A, B, corr=False, origscm=False):
    """
    Spectral Correlation Mapper, compute similarity between two spectra using
    Pearsonian correlation coefficient. Value varies between -1 and 1.

    Parameters:
    - `A`: reference matrix, of dimension rowxcolxwavelength.
    - `B`: target matrix, of dimension rowxcolxwavelength.

    Return: distance matrix, of dimension rowxcol
    """
    wvl = A.shape[2]
    A_bar = np.average(A, 2)
    B_bar = np.average(B, 2)
    A_bar_t = np.tile(A_bar[:, :, np.newaxis], (1, wvl))
    B_bar_t = np.tile(B_bar[:, :, np.newaxis], (1, wvl))
    A_cent = A - A_bar_t
    B_cent = B - B_bar_t
    divident = np.sum(np.multiply(A_cent, B_cent), axis=2)
    divisor = np.multiply(
        np.sqrt(np.sum(np.power(A_cent, 2), axis=2)),
        np.sqrt(np.sum(np.power(B_cent, 2), axis=2)))
    if corr:
        return 1 - \
            (1 + np.around(np.divide(divident, divisor), decimals=4)) / 2.0
    elif not corr and not origscm:
        return 1 - np.around(np.divide(divident, divisor), decimals=4)
    elif not corr and origscm:
        return np.around(np.divide(divident, divisor), decimals=4)
