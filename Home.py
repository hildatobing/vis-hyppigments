#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:04:28 2023

@author: hildad
"""
from glob import glob
from math import isnan
from matplotlib import pyplot as plt

import distance_functions as dist
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import spectral as sp
import streamlit as st

from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode


st.set_page_config(
    page_title='Hyperspectral Pigments', 
    page_icon='🎨',
    layout='wide', 
    initial_sidebar_state='expanded')


def sidebar_layout(list_mode):
    st.selectbox('Select mode', list_mode, key='mode_selector')
    if st.session_state.mode_selector == list_mode[2]:
        file_help = ':red-background[**File structure requirements**]\n - :red'\
            '-background[The first column (row 2 onwards) must contain the wav'\
            'elengths in nanometer unit. Write \"wvl\" on the first row.]\n - '\
            ':red-background[Each column represents each spectrum, with the fi'\
            'rst row being the spectrum/pigment name.]\n - :red-background[The'\
            ' name of the spectra should not contain neither a comma nor a sem'\
            'icolon.]'
        st.file_uploader(
            'Select .csv file', accept_multiple_files=False, 
            key='file_selector', help=file_help)
    st.write('#')

    st.header('About')
    st.markdown(
        'Hyperspectral dataset made of pure pigments from Kremer color charts.'\
        ' The full dataset is also available via [Zenodo](https://doi.org/10.5'\
        '281/zenodo.5592484).')
    st.markdown(
        'The color of each spectrum in a plot is not randomly picked, but gene'\
        'rated using CIE 1931 Color Matching Function 2&deg; Standard Observer'\
        ' and D65 Standard Illuminant, to simulate human perception of the pig'\
        'ment under daylight at noon.')
    st.markdown(
        '📨 <a href="mailto:hilda.deborah@ntnu.no">Send your feedback</a>',
        unsafe_allow_html=True)
    
    st.header('Citation')
    st.success(
        'H. Deborah, \"Hyperspectral Pigment Dataset,\" 2022 12th Workshop on '\
        'Hyperspectral Imaging and Signal Processing: Evolution in Remote Sens'\
        'ing (WHISPERS), Rome, Italy, 2022, pp.1-5, DOI: [10.1109/WHISPERS5617'\
        '8.2022.9955067](https://doi.org/10.1109/WHISPERS56178.2022.9955067).')
    
    st.subheader('Version history')
    st.markdown(
        '**2.0** Pigment matcher added<br>'\
        '**1.0** Explorer modes: Single pigments and comparison', 
        unsafe_allow_html=True)


def get_pname(pid, df):
    res = df.loc[df['fid'] == pid]
    return res.iloc[0][1]


def get_sli():
    f = 'Data/__speclib_averages.hdr'
    sli = sp.envi.open(f)
    pnames = sli.names
    wvl = sli.bands.centers

    fc = 'Data/__colorlib_averages.pickle'
    cols = pd.read_pickle(fc)

    return sli, wvl, pnames, cols


def matcher_plot(
        twvl, spectra, labels, matches_colors=None, first_render=False):
    
    nmatches = len(labels) - 1
    fig = go.Figure()
    if not first_render:
        fig.add_trace(go.Scatter(
            x=twvl, y=spectra[0, :], name='Target: '+labels[0], 
            line=dict(color='red', dash='dot', width=3)))
        if nmatches > 0:
            for i in range(1, nmatches+1):
                c = 'rgb(' + ','.join(str(x) for x in matches_colors[i-1]) + ')'
                fig.add_trace(go.Scatter(
                    x=twvl, y=spectra[i, :], name=labels[i], 
                    line=dict(color=c, width=2)))

    fig.update_layout(
        xaxis_title='Wavelength (in nanometer)',
        yaxis_title='Reflectance',
        legend_title='Legend',
        # legend=dict(
        #     orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
    )
    fig.update_yaxes(range=[-.01, 1.01])
    st.plotly_chart(fig)


def explorer_plot(df, first_render=False, mode='single'):
    sli, wvl, pnames, cols = get_sli()

    spectra = [wvl]
    pids = []
    fig = go.Figure()
    if not first_render:
        if mode == 'single':
            name = df.iloc[0]['Pigment name'] + ', shade-'
            name = name[0].upper() + name[1:]
            for i in range(1, 5):
                pid = df.iloc[0]['fid'] + '_sh' + str(i)
                col = 'rgb(' + ','.join(str(x) for x in cols[pid]) + ')'
                fig.add_trace(go.Scatter(
                    x=wvl, y=sli.spectra[pnames.index(pid), :], 
                    name=name+str(i), line=dict(color=col,width=3)))
                spectra.append(sli.spectra[pnames.index(pid), :])
                pids.append(df.iloc[0]['Pigment number'] + ' sh-' + str(i))
        elif mode == 'compare':
            for sel_row in df:
                name = sel_row['Pigment name']
                pid = sel_row['fid'] + '_sh4'
                col = 'rgb(' + ','.join(str(x) for x in cols[pid]) + ')'
                fig.add_trace(go.Scatter(
                    x=wvl, y=sli.spectra[pnames.index(pid), :], 
                    name=name, line=dict(color=col,width=3)))
                spectra.append(sli.spectra[pnames.index(pid), :])
                pids.append(sel_row['Pigment number'])
            fig['data'][0]['showlegend'] = True

    fig.update_layout(
        xaxis_title='Wavelength (in nanometer)',
        yaxis_title='Reflectance',
        legend_title='Pigment name')
    fig.update_yaxes(range=[-.01, 1.01])
    st.plotly_chart(fig)
    
    hdr = ['Wavelength'] + pids
    data = pd.DataFrame.from_dict(
        dict(zip(hdr, spectra))).to_csv(index=False).encode('utf-8')
    return hdr, data


def single_mode(df):
    # Configure grid options
    builder = GridOptionsBuilder.from_dataframe(df.iloc[:, :3])
    builder.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
    builder.configure_selection(selection_mode='single', use_checkbox=False)
    grid_options = builder.build()

    grid_table = AgGrid(
        df.iloc[:, [0,1,2,5]], gridOptions=grid_options, 
        enable_enterprise_modules=False,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW)
    if grid_table['selected_rows']:
        pid = grid_table['selected_rows'][0]['fid']
        sel_pigments = df[df['fid'] == pid]
        
        _, data = '', None
        with plot_area:
            _, data = explorer_plot(
                sel_pigments, first_render=False, mode='single')
        with download_button_area:
            st.download_button(
                label='Download spectra as CSV', data=data, 
                file_name='spectra.csv', mime='text/csv')


def compare_mode(df):
    # Configure grid options
    builder = GridOptionsBuilder.from_dataframe(df.iloc[:, :3])
    builder.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
    builder.configure_selection(selection_mode='multiple', use_checkbox=True)
    grid_options = builder.build()

    grid_table = AgGrid(
        df.iloc[:, [0,1,2,5]], gridOptions=grid_options, 
        enable_enterprise_modules=False, 
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW)
    if grid_table['selected_rows']:
        _, data = '', None
        with plot_area:
            _, data = explorer_plot(
                grid_table['selected_rows'], first_render=False, mode='compare')
        
        with download_button_area:
            st.download_button(
                label='Download spectra as CSV', data=data, 
                file_name='spectra.csv', mime='text/csv')


if __name__ == '__main__':
    
    # Keys initialisation
    list_mode = [
        'Explorer - Single pigments', 'Explorer - Comparison', 
        'Pigment matcher']
    if 'mode_selector' not in st.session_state:
        st.session_state['mode_selector'] = list_mode[0]
    
    with st.sidebar:
        st.title('Hyperspectral Pigment Dashboard 2.0 🎨')
        sidebar_layout(list_mode)
        
    pagetitle = st.empty()
    general_instr_area = st.container()
    general_selector_area = st.container()
    plot_area = st.empty()
    download_button_area = st.empty()
    table_instr_area = st.empty()
    table_area = st.empty()
    

    f = 'Data/__pigmentlist.xls'
    df = pd.read_excel(f)
    df['pnum'] = df['pnum'].astype(str)
    df = df.rename(columns={
        'pnum':'Pigment number', 'pname':'Pigment name', 'pdesc':'Description'})
    if st.session_state.mode_selector == list_mode[0]:
        pagetitle.header(list_mode[0].replace(' -', ':'))
        with table_area:
            st.markdown(
                '<b><u>Select one pigment from the table below</u></b> to show'\
                ' its reflectance spectra in the interactive plot, where you w'\
                'ould also be able to download the plot. Note that for each pi'\
                'gment, four spectra will be shown, each corresponding to the '\
                'different shades of the printed Kremer card.',
                unsafe_allow_html=True)
        with plot_area:
            explorer_plot(df, first_render=True)
        single_mode(df)
    
    elif st.session_state.mode_selector == list_mode[1]:
        pagetitle.header(list_mode[1].replace(' -', ':'))
        with table_area:
            st.markdown(
                '<b><u>Select multiple pigments from the table below (use the '\
                'tick boxes)</u></b> to show their reflectance spectra in the '\
                'interactive plot. Unlike the Explore Pigment mode, only one s'\
                'pectrum from each selected pigment will be shown.', 
                unsafe_allow_html=True)
        with plot_area:
            explorer_plot(df, first_render=True)
        compare_mode(df)

    elif st.session_state.mode_selector == list_mode[2]:
        pagetitle.header(list_mode[2])
        dabbrv, dnames = zip(*dist.get_distance_dict().items())

        with general_instr_area:
            st.markdown(
                'This pigment matcher only accepts and assumes input data with'\
                'in the range of 400-1000 nanometers, and of reflectance value'\
                's between 0 and 1. If your data exceeds the wavelength range,'\
                ' it will be automatically clipped. <b><u>Use the sidebar to u'\
                'pload your spectra (.csv)</u>.</b> Click the tooltip/ questio'\
                'n mark next to the uploader to see how you should structure y'\
                'our file.', unsafe_allow_html=True)
            
        with general_selector_area:
            if st.session_state.file_selector is not None:
                sli, sliwvl, pnames, slicolors = get_sli()
                
                input_data = pd.read_csv(st.session_state.file_selector)
                input_cols = list(input_data)
                wvl = input_data[input_cols[0]].values
                
                c1, c2 = st.columns([1, 2], gap='medium')
                c1.selectbox(
                    'Select target to match', options=input_cols[1:], 
                    key='selected_target')
                c2.selectbox(
                    'Select a distance function to use', options=dnames,
                    key='selected_distfunction')
                # st.checkbox(
                #     'First derivative?', key='selected_derivative')
                st.slider(
                    'Select number of matches to return', 1, 10, 3, 
                    key='selected_nmatches')
                
                target_spectrum = np.array(
                    input_data[st.session_state.selected_target])
                interp_spectrum, slispectra, sliwvl = dist.interpolate(
                    target_spectrum, wvl, sli.spectra, sliwvl)

                dfun = dabbrv[
                    dnames.index(st.session_state.selected_distfunction)]

                dvals = dist.get_distance_values(
                    interp_spectrum, slispectra, fun=dfun).flatten()
                asc_idx = np.argsort(dvals)
                
                labels = [st.session_state.selected_target]
                colors = []
                spectra = interp_spectrum.reshape(1, len(sliwvl))
                for i in range(st.session_state.selected_nmatches):
                    idx = asc_idx[i]
                    fid = pnames[idx]
                    label = get_pname(fid[:-4], df) + '; d=%.3f' %(dvals[idx])
                    labels.append(label)
                    colors.append(slicolors[fid])
                    spectra = np.concatenate((
                        spectra, slispectra[idx].reshape(1, len(sliwvl))), 
                        axis=0)
                matcher_plot(
                    wvl, spectra, labels, matches_colors=colors, 
                    first_render=False)