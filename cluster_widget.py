import ipywidgets as widgets
from ipywidgets import Checkbox, VBox, HBox, HTML, Dropdown, Image, IntSlider
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nlp_suite.clustering.utils import plot_3d_clusters, classify_text
from nlp_suite.text_analysis.text_analysis_kevin import *

# widgets for the dashboard follow a similar pattern, as follows
# note the get_widget and init_widget_data methods.
class ClusterWidget():

    # you can setup the main skeleton of the widget here as instance variables, such 
    # that its elements can be updated when new data comes in.
    # however, it should not rely on the user_info yet, because that is not available when the app first starts.
    def __init__(self):
        
        pkl_data = pickle.load(open('./nlp_suite/clustering/cluster_data.pkl', 'rb'))
        labels, pca = pkl_data['labels'], pkl_data['pca']
        self.fig = plot_3d_clusters(pca, labels, max_points=50000)
        self.cluster_dd = Dropdown(
                options=[(f'Cluster {i}', i) for i in range(max(labels)+1)],
                value=0,
                description='Cluster:',
            )
        def on_click(trace, points, state):
            self.cluster_dd.value = int(trace.marker.color[points.point_inds[0]])
            
        self.fig.data[0].on_click(on_click)
        layout = widgets.Layout(width='auto')
        self.header = HTML(description="3D visualization of user clusters",layout=layout)
        self.widget = VBox([self.header, self.fig, self.cluster_dd])


    # returns the widget skeleton. this is used when the dashboard is first displayed since 
    # Voila cannot re-display new elements after its initial loading in the browser.
    def get_widget(self):
        return self.widget


    # called by the dashboard when a new chat log is uploaded
    # the new user_info is passed, and the skeleton can be updated to reflect it.
    def init_widget_data(self, user_info):
        self.widget.description = user_info["user_name"]

    def checkboxes(self):
        user_messages_path='./cached_user_data/muffins/user_messages.p'
        df = pd.DataFrame()
        df['user_messages'] = pickle.load(open(user_messages_path, "rb"))
        self.text_analysis = get_text_analysis(df)
        get_plots(*self.text_analysis)
    
        self.Graphs = [
            'Message Length Histogram',
            'Average Word Length Histogram',
            'Top 10 Stop Words Histogram',
            'Top 10 Unigrams Histogram',
            'Top 10 Bigrams Histogram',
            'Top 10 Trigrams Histogram',
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
        self.plots_list = [0,0,0,0,0,0,0]
        self.widgets_dict = {}
        self.slider = IntSlider(
            value=300,
            min=100,
            max=1000,
            step=10,
            description='Size Of Plots:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )

        self.slider_widget = widgets.interactive(self.set_plot_sizes,x=self.slider)
        

        self.vb = VBox([self.slider_widget,
            HBox([self.cb_all, self.cb0, self.cb1, self.cb2]), 
            HBox([self.cb3, self.cb4, self.cb5, self.cb6])
            ])

        self.widget.children = VBox([self.header, self.fig, self.cluster_dd, self.vb])

        self.checkboxes = self.vb.children
        
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
        if not button['new']:
            self.cb3.value=False
            self.cb0.value=False
            self.cb1.value=False
            self.cb2.value=False
            self.cb4.value=False
            self.cb5.value=False
            self.cb6.value=False
        else:
            self.cb0.value=True
            self.cb1.value=True
            self.cb2.value=True
            self.cb3.value=True
            self.cb4.value=True
            self.cb5.value=True
            self.cb6.value=True

    def cb0_show(self,button):
        key = 'plot0'
        if button['new']:
            self.widgets_dict[key] = self.plots_list[0]
            self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb1_show(self,button):
        key = 'plot1'
        if button['new']:
            self.widgets_dict[key] = self.plots_list[1]
            self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb2_show(self,button):
        key = 'plot2'
        if button['new']:
            self.widgets_dict[key] = self.plots_list[2]
            self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb3_show(self,button):
        
        key = 'plot3'
        if button['new']:
            self.widgets_dict[key] = self.plots_list[3]
            self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb4_show(self,button):
        key = 'plot4'
        if button['new']:
            self.widgets_dict[key] = self.plots_list[4]
            self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb5_show(self,button):
        key = 'plot5'
        if button['new']:
            self.widgets_dict[key] = self.plots_list[5]
            self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
            except KeyError:
                pass

    def cb6_show(self,button):
        key = 'plot6'
        if button['new']:
            self.widgets_dict[key] = self.plots_list[6]
            self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
        else:
            try:
                del self.widgets_dict[key]
                self.vb.children = [*self.checkboxes,*self.widgets_dict.values()]
            except KeyError:
                pass
    def set_plot_sizes(self,x):
        for i in range(len(self.Graphs)):
            plot = open('plots/'+self.plot_names[i], "rb")
            image = plot.read()
            self.plots_list[i] = Image(value=image,format='png',width=x)
        self.cb0_show({'new':self.cb0.value})
        self.cb1_show({'new':self.cb1.value})
        self.cb2_show({'new':self.cb2.value})
        self.cb3_show({'new':self.cb3.value})
        self.cb4_show({'new':self.cb4.value})
        self.cb5_show({'new':self.cb5.value})
        self.cb6_show({'new':self.cb6.value})


    

