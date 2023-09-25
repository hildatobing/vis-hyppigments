#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:04:28 2023

@author: hildad
"""
from glob import glob
from math import isnan
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import spectral as sp
import streamlit as st

from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode

hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
}
</style>
"""

def get_pname(pid, df):
    res = df.loc[df['fid'] == pid]
    return res.iloc[0][1]


def plot(df, first_render=False, mode='single'):
    f = 'Data/__speclib_averages.hdr'
    sli = sp.envi.open(f)
    pnames = sli.names
    wvl = sli.bands.centers

    fc = 'Data/__colorlib_averages.pickle'
    cols = pd.read_pickle(fc)

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
        elif mode == 'compare':
            for sel_row in df:
                name = sel_row['Pigment name']
                pid = sel_row['fid'] + '_sh4'
                col = 'rgb(' + ','.join(str(x) for x in cols[pid]) + ')'
                fig.add_trace(go.Scatter(
                    x=wvl, y=sli.spectra[pnames.index(pid), :], 
                    name=name, line=dict(color=col,width=3)))
            fig['data'][0]['showlegend'] = True

    fig.update_layout(
        xaxis_title='Wavelength (in nanometer)',
        yaxis_title='Reflectance',
        legend_title='Pigment name'
    )
    fig.update_yaxes(range=[-.01, 1.01])
    st.plotly_chart(fig)


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
        with plot_area:
            plot(sel_pigments, first_render=False, mode='single')


def compare_mode(df):
    # Configure grid options
    builder = GridOptionsBuilder.from_dataframe(df.iloc[:, :3])
    builder.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
    builder.configure_selection(selection_mode='multiple', use_checkbox=True)
    grid_options = builder.build()

    grid_table = AgGrid(
        df.iloc[:, [0,1,2,5]], gridOptions=grid_options, 
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW)
    if grid_table['selected_rows']:
        with plot_area:
            plot(grid_table['selected_rows'], first_render=False, mode='compare')


st.set_page_config(page_title="Hyperspectral Pigments")
if __name__ == '__main__':
    st.title('Hyperspectral Pigments')
    # Hiding the default top-right menu from streamlit
    st.markdown(hide_menu, unsafe_allow_html=True)

    st.markdown(
        'Hyperspectral dataset made of pure pigments from Kremer color charts. If'\
        ' you use this platform and/or the dataset, please cite the following art'\
        'icle:')
    st.markdown(
        '<sup>H. Deborah, \"Hyperspectral Pigment Dataset,\" 2022 12th Workshop o'\
        'n Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensi'\
        'ng (WHISPERS), Rome, Italy, 2022, pp.1-5, doi: 10.1109/WHISPERS56178.202'\
        '2.9955067.</sup>', unsafe_allow_html=True)
    st.write(
        'The full dataset is also available via '\
        '[Zenodo](https://doi.org/10.5281/zenodo.5592484).')


    st.header('Explorer')
    modes = ('Single pigment', 'Comparison')
    mode = st.selectbox('Choose mode:', modes)

    instr_area = st.empty()
    plot_area = st.empty()
    table_instr_area = st.empty()
    table_area = st.empty()

    f = 'Data/__pigmentlist.xls'
    df = pd.read_excel(f)
    df['pnum'] = df['pnum'].astype(str)
    df = df.rename(columns={'pnum':'Pigment number', 'pname':'Pigment name', \
                            'pdesc':'Description'})

    if mode == modes[0]:
        with instr_area:
            st.markdown(
                '<sub><b><u>Select one pigment from the table below</u></b> to show i'\
                'ts reflectance spectra in the interactive plot, where you would also'\
                ' be able to download the plot. Note that for each pigment, four spec'\
                'tra will be shown, each corresponding to the different shades of the'\
                ' printed Kremer card. Colors of each spectrum is generated using CIE'\
                ' 1931 Color Matching Function 2&deg; Standard Observer and D65 Stand'\
                'ard Illuminant, to simulate human perception of the pigment under da'\
                'ylight at noon.</sub>', unsafe_allow_html=True)
        # Show plot container
        with plot_area:
            plot(df, first_render=True)
        single_mode(df)
    elif mode == modes[1]:
        with instr_area:
            st.markdown(
                '<sub><b><u>Select multiple pigments from the table below (use the '\
                'tick boxes)</u></b> to show their reflectance spectra in the inter'\
                'active plot. Unlike the single pigment mode, only one spectrum fro'\
                'm each selected pigment will be shown. The color of each spectrum '\
                'is generated using CIE  1931 Color Matching Function 2&deg; Standa'\
                'rd Observer and D65 Standard Illuminant, to simulate human percept'\
                'ion of the pigment under daylight at noon.</sub>', 
                unsafe_allow_html=True)
        with plot_area:
            plot(df, first_render=True)
        compare_mode(df)
