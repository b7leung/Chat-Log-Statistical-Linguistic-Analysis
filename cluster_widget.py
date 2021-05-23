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

        self.hi=False
        self.here = False
        self.here2 = False
        self.here3 = False
        user_messages_path='/Users/kevinyoussef/Desktop/ECE 229/Project/ece_229_group_9_nlp_suite/cached_user_data/Iinden/user_messages.p'
        df = pd.DataFrame()
        df['user_messages'] = pickle.load(open(user_messages_path, "rb"))
        self.text_analysis = get_text_analysis(df)
        get_plots(*self.text_analysis)
        
        Graphs = [
            'Message Length Histogram',
            'Average Word Length Histogram',
            'Top 10 Stop Words Histogram',
            'Top 10 Unigrams Histogram',
            'Top 10 Bigrams Histogram',
            'Top 10 Trigrams Histogram',
            'Word Cloud'
        ]

        plot_names = [
            'message_lengths_hist.png',
            'average_word_lengths_hist.png',
            'stop_dic_histogram.png',
            'unigrams.png',
            'bigrams.png',
            'trigrams.png',
            'word_cloud.png',
        ]

        self.cb_all = Checkbox(description='All Graphs')
        self.cb0 = Checkbox(description=Graphs[0])
        self.cb1 = Checkbox(description=Graphs[1])
        self.cb2 = Checkbox(description=Graphs[2])
        self.cb3 = Checkbox(description=Graphs[3])
        self.cb4 = Checkbox(description=Graphs[4])
        self.cb5 = Checkbox(description=Graphs[5])
        self.cb6 = Checkbox(description=Graphs[6])
        self.plots_list = [0,0,0,0,0,0,0]
        widgets_dict = {}

        def cb_all_show(button):
            if not button['new']:
                self.hi = True
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
                self.here3 = True

        def cb0_show(button):
            key = 'plot0'
            if button['new']:
                widgets_dict[key] = self.plots_list[0]
                self.vb.children = [*self.checkboxes,*widgets_dict.values()]
            else:
                try:
                    del widgets_dict[key]
                    self.vb.children = [*self.checkboxes,*widgets_dict.values()]
                except KeyError:
                    print("ERROR")
                    pass

        def cb1_show(button):
            key = 'plot1'
            if button['new']:
                widgets_dict[key] = self.plots_list[1]
                self.vb.children = [*self.checkboxes,*widgets_dict.values()]
            else:
                try:
                    del widgets_dict[key]
                    self.vb.children = [*self.checkboxes,*widgets_dict.values()]
                except KeyError:
                    print("ERROR")
                    pass

        def cb2_show(button):
            key = 'plot2'
            if button['new']:
                widgets_dict[key] = self.plots_list[2]
                self.vb.children = [*self.checkboxes,*widgets_dict.values()]
            else:
                try:
                    del widgets_dict[key]
                    self.vb.children = [*self.checkboxes,*widgets_dict.values()]
                except KeyError:
                    print("ERROR")
                    pass

        def cb3_show(button):
            
            key = 'plot3'
            if button['new']:
                widgets_dict[key] = self.plots_list[3]
                self.vb.children = [*self.checkboxes,*widgets_dict.values()]
                self.here = True
            else:
                try:
                    del widgets_dict[key]
                    self.vb.children = [*self.checkboxes,*widgets_dict.values()]
                except KeyError:
                    print("ERROR")
                    pass

        def cb4_show(button):
            key = 'plot4'
            if button['new']:
                widgets_dict[key] = self.plots_list[4]
                self.vb.children = [*self.checkboxes,*widgets_dict.values()]
            else:
                try:
                    del widgets_dict[key]
                    self.vb.children = [*self.checkboxes,*widgets_dict.values()]
                except KeyError:
                    print("ERROR")
                    pass

        def cb5_show(button):
            key = 'plot5'
            if button['new']:
                widgets_dict[key] = self.plots_list[5]
                self.vb.children = [*self.checkboxes,*widgets_dict.values()]
            else:
                try:
                    del widgets_dict[key]
                    self.vb.children = [*self.checkboxes,*widgets_dict.values()]
                except KeyError:
                    print("ERROR")
                    pass

        def cb6_show(button):
            key = 'plot6'
            if button['new']:
                widgets_dict[key] = self.plots_list[6]
                self.vb.children = [*self.checkboxes,*widgets_dict.values()]
            else:
                try:
                    del widgets_dict[key]
                    self.vb.children = [*self.checkboxes,*widgets_dict.values()]
                except KeyError:
                    print("ERROR")
                    pass

        def set_plot_sizes(x):
            for i in range(len(Graphs)):
                self.here2 = True
                plot = open(plot_names[i], "rb")
                image = plot.read()
                self.plots_list[i] = Image(value=image,format='png',width=x)
            cb0_show({'new':self.cb0.value})
            cb1_show({'new':self.cb1.value})
            cb2_show({'new':self.cb2.value})
            cb3_show({'new':self.cb3.value})
            cb4_show({'new':self.cb4.value})
            cb5_show({'new':self.cb5.value})
            cb6_show({'new':self.cb6.value})

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

        self.vb = VBox([self.slider,
            HBox([self.cb_all, self.cb0, self.cb1, self.cb2]), 
            HBox([self.cb3, self.cb4, self.cb5, self.cb6])
            ])
        self.checkboxes = self.vb.children

        self.slider.observe(set_plot_sizes, names='values')
        self.cb_all.observe(cb_all_show, names='value')
        self.cb0.observe(cb0_show, names='value')
        self.cb1.observe(cb1_show, names='value')
        self.cb2.observe(cb2_show, names='value')
        self.cb3.observe(cb3_show, names='value')
        self.cb4.observe(cb4_show, names='value')
        self.cb5.observe(cb5_show, names='value')
        self.cb6.observe(cb6_show, names='value')

        layout = widgets.Layout(width='auto')
        header = HTML(description="3D visualization of user clusters",layout=layout)
        self.widget = VBox([header, self.fig, self.cluster_dd, self.vb])

    


    # returns the widget skeleton. this is used when the dashboard is first displayed since 
    # Voila cannot re-display new elements after its initial loading in the browser.
    def get_widget(self):
        return self.widget


    # called by the dashboard when a new chat log is uploaded
    # the new user_info is passed, and the skeleton can be updated to reflect it.
    def init_widget_data(self, user_info):
        self.widget.description = user_info["user_name"]

