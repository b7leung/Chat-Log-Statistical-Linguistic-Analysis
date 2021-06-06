import ipywidgets as widgets
from ipywidgets import Checkbox, VBox, HBox, HTML, Dropdown, Image, IntSlider, Valid, Box, Layout
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from nlp_suite.clustering.utils import plot_3d_clusters, classify_text
from nlp_suite.text_analysis.text_analysis_kevin import *


class ClusterWidget():
    '''Widget containing clustering analysis
    '''    
    def __init__(self):
        '''Constructor Method
        '''        
        self.Graphs = [
            'Message Length',
            'Average Word Length',
            'Top 10 Stop Words',
            'Top 10 Unigrams',
            'Top 10 Bigrams',
            'Top 10 Trigrams',
            'Word Cloud'
        ]

        self.plot_names = [
        'message_lengths_hist.png',
        'average_word_lengths_hist.png',
        'stop_dic_histogram.png',
        'unigrams.png',
        'bigrams.png',
        'trigrams.png',
        'word_cloud.png',
    ]
        
        self.cb_all = Checkbox(description='All Graphs')
        self.cb0 = Checkbox(description=self.Graphs[0])
        self.cb1 = Checkbox(description=self.Graphs[1])
        self.cb2 = Checkbox(description=self.Graphs[2])
        self.cb3 = Checkbox(description=self.Graphs[3])
        self.cb4 = Checkbox(description=self.Graphs[4])
        self.cb5 = Checkbox(description=self.Graphs[5])
        self.cb6 = Checkbox(description=self.Graphs[6])
        self.cluster_plots_list = [0,0,0,0,0,0,0]
        self.user_plots_list = [0,0,0,0,0,0,0]
        self.widgets_dict = {}

        self.slider = IntSlider(
            value=100,
            min=100,
            max=800,
            step=10,
            description='Size Of Plots:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )

        self.slider_widget = widgets.interactive(self.set_plot_sizes,x=self.slider)     




        self.data_processed_bool=False
        pkl_data = pickle.load(open('./nlp_suite/clustering/cluster_data.pkl', 'rb'))
        self.labels, pca = pkl_data['labels'], pkl_data['pca']
        self.encodings = pickle.load(open('./nlp_suite/Clustering/encodings.pkl', 'rb'))
        self.fig = plot_3d_clusters(pca, self.labels, max_points=50000)
        self.cluster_dd = Dropdown(
                options=[(f'Cluster {i}', i) for i in range(max(self.labels)+1)],
                value=0,
                description='Cluster:',
            )
        def on_click(trace, points, state):
            self.cluster_dd.value = int(trace.marker.color[points.point_inds[0]])
            if self.data_processed_bool:
                self.set_plot_sizes(self.slider.value)
                # self.slider.value = self.slider.value
            
        

        self.fig.data[0].on_click(on_click)
        layout = widgets.Layout(width='auto')
        self.header = HTML(description="3D visualization of user clusters",layout=layout)
        self.processed_data = Valid(value=False,description='Data Processed')
        self.vb = VBox([self.processed_data])
        self.widget = VBox([self.header, self.fig, self.cluster_dd, self.vb])
        

    # returns the widget skeleton. this is used when the dashboard is first displayed since 
    # Voila cannot re-display new elements after its initial loading in the browser.
    def get_widget(self):
        '''Obtain underlying widget object

        :return: Widget containing clustering data
        :rtype: ipywidgets.VBox 
        '''        
        return self.widget

    def data_been_processed(self,user_info):
        '''Process user information

        :param user_info: Dictionary containing messages for each user
        :type user_info: dict
        '''        
        self.vb.children = [HTML('Processing ...')]

        user_messages_path='./cached_user_data/'+user_info['user_name']+'/user_messages.p'
        self.df = pd.DataFrame()
        self.df['user_messages'] = pickle.load(open(user_messages_path, "rb"))
        self.text_analysis = get_text_analysis(self.df)
        get_plots(*self.text_analysis)
        
        self.checkboxes()
        self.slider.value = 300
        self.data_processed_bool=True

    def checkboxes(self):
        '''[summary]
        '''

        # self.df = pd.DataFrame()
        # message_file_path = '../user_chat_dataframe.pkl'
        # if os.path.exists(message_file_path):
        #     self.df['user_messages'] = [x for sublist in pickle.load(open(message_file_path, "rb"))['Chats'].iloc[np.where(self.labels==4)[0][:10000]] for x in sublist]
        # else:
        #     user_messages_path='./cached_user_data/'+user_info['user_name']+'/user_messages.p'
        #     self.df['user_messages'] = pickle.load(open(user_messages_path, "rb"))
        # self.text_analysis = get_text_analysis(self.df)
        # get_plots(*self.text_analysis)

        
    
           

        self.vb.children = [VBox([self.slider_widget,
            HBox_space([self.cb_all, self.cb0, self.cb1, self.cb2]), 
            HBox_space([self.cb3, self.cb4, self.cb5, self.cb6]),
            HBox_space([HTML('<p style="font-size:40px"><b>Cluster</b></p>'), HTML('<p style="font-size:40px"><b>User</b></p>')],layout=Layout(height='40px'))
            ])]

        self.checkboxes_widgets = self.vb.children
        
        self.slider.observe(self.set_plot_sizes, names='values')
        self.cb_all.observe(self.cb_all_show, names='value')
        self.cb0.observe(self.cb0_show, names='value')
        self.cb1.observe(self.cb1_show, names='value')
        self.cb2.observe(self.cb2_show, names='value')
        self.cb3.observe(self.cb3_show, names='value')
        self.cb4.observe(self.cb4_show, names='value')
        self.cb5.observe(self.cb5_show, names='value')
        self.cb6.observe(self.cb6_show, names='value')

            
    def cb_all_show(self,button):
        if button['new']:
            self.cb0.value=True
            self.cb1.value=True
            self.cb2.value=True
            self.cb3.value=True
            self.cb4.value=True
            self.cb5.value=True
            self.cb6.value=True
        else:
            self.cb0.value=False
            self.cb1.value=False
            self.cb2.value=False
            self.cb3.value=False
            self.cb4.value=False
            self.cb5.value=False
            self.cb6.value=False

    def cb0_show(self,button):
        key = 'plot0'
        if button['new']:
            self.widgets_dict[key] = HBox_space([self.cluster_plots_list[0],self.user_plots_list[0]])
            self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb1_show(self,button):
        key = 'plot1'
        if button['new']:
            self.widgets_dict[key] = HBox_space([self.cluster_plots_list[1],self.user_plots_list[1]])
            self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb2_show(self,button):
        key = 'plot2'
        if button['new']:
            self.widgets_dict[key] = HBox_space([self.cluster_plots_list[2],self.user_plots_list[2]])
            self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb3_show(self,button):
        
        key = 'plot3'
        if button['new']:
            self.widgets_dict[key] = HBox_space([self.cluster_plots_list[3],self.user_plots_list[3]])
            self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb4_show(self,button):
        key = 'plot4'
        if button['new']:
            self.widgets_dict[key] = HBox_space([self.cluster_plots_list[4],self.user_plots_list[4]])
            self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb5_show(self,button):
        key = 'plot5'
        if button['new']:
            self.widgets_dict[key] = HBox_space([self.cluster_plots_list[5],self.user_plots_list[5]])
            self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb6_show(self,button):
        key = 'plot6'
        if button['new']:
            self.widgets_dict[key] = HBox_space([self.cluster_plots_list[6],self.user_plots_list[6]])
            self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes_widgets,*self.widgets_dict.values()]
            except KeyError:
                pass

    def set_plot_sizes(self,x):
        for i in range(len(self.Graphs)):
            plot = open('cluster_'+str(self.cluster_dd.value)+'_plots/'+self.plot_names[i], "rb")
            image = plot.read()
            self.cluster_plots_list[i] = Image(value=image,format='png',width=x)
            plot = open('plots/'+self.plot_names[i], "rb")
            image = plot.read()
            self.user_plots_list[i] = Image(value=image,format='png',width=x)
        self.cb0_show({'new':self.cb0.value})
        self.cb1_show({'new':self.cb1.value})
        self.cb2_show({'new':self.cb2.value})
        self.cb3_show({'new':self.cb3.value})
        self.cb4_show({'new':self.cb4.value})
        self.cb5_show({'new':self.cb5.value})
        self.cb6_show({'new':self.cb6.value})

def HBox_space(*pargs, **kwargs):
    '''Displays multiple widgets horizontally using the flexible box model.

    :return: [description]
    :rtype: [type]
    '''    
    box = Box(*pargs, **kwargs)
    box.layout.display = 'flex'
    box.layout.align_items = 'stretch'
    box.layout.justify_content = 'space-around'
    return box


    

