import ipywidgets as widgets
import pickle
from nlp_suite.clustering.utils import plot_3d_clusters, classify_text
from nlp_suite.text_analysis.text_analysis import frequency_info, generate_word_cloud, plot_sentence_length_histogram
from IPython.display import display
# widgets for the dashboard follow a similar pattern, as follows
# note the get_widget and init_widget_data methods.
class ClusterWidget():

    # you can setup the main skeleton of the widget here as instance variables, such 
    # that its elements can be updated when new data comes in.
    # however, it should not rely on the user_info yet, because that is not available when the app first starts.
    def __init__(self):
        self.widget = widgets.Text(description="3D visualization of user clusters")
        pkl_data = pickle.load(open('./nlp_suite/clustering/cluster_data.pkl', 'rb'))
        labels, pca = pkl_data['labels'], pkl_data['pca']
        self.fig = plot_3d_clusters(pca, labels, max_points=50000)
        self.cluster_dd = widgets.Dropdown(
                options=[(f'Cluster {i}', i) for i in range(max(labels)+1)],
                value=0,
                description='Cluster:',
            )
        def on_click(trace, points, state):
            self.cluster_dd.value = int(trace.marker.color[points.point_inds[0]])
        
        
        self.output1 = widgets.Output()
        @self.output1.capture(clear_output=True)
        def on_click2(b):
            with self.output1:
                return frequency_info(user_chat_df.loc[labels == self.cluster_dd.value])


        self.output2 = widgets.Output()
        @self.output2.capture(clear_output=True)
        def on_click3(b):
            with self.output2:
                return generate_word_cloud(user_chat_df.loc[labels == self.cluster_dd.value])
        
    
        self.fig.data[0].on_click(on_click)

        self.output3 = widgets.Output()
        @self.output3.capture(clear_output=True)
        def on_click4(b):
            with self.output2:
                return plot_sentence_length_histogram(user_chat_df.loc[labels == self.cluster_dd.value])
        
    
        self.fig.data[0].on_click(on_click)
        

        user_chat_df = pickle.load(open('./nlp_suite/clustering/user_chat_dataframe.pkl', 'rb'))


        self.generate_plots = widgets.Button(description = 'Generate Analysis')
        self.generate_plots.on_click(on_click2)
        self.generate_plots.on_click(on_click3)
        self.generate_plots.on_click(on_click4)

        self.Vbox = widgets.VBox(children = [self.output1, self.output2])


        self.widget = widgets.VBox([self.fig, self.cluster_dd, self.generate_plots, self.Vbox
        ])


    # returns the widget skeleton. this is used when the dashboard is first displayed since 
    # Voila cannot re-display new elements after its initial loading in the browser.
    def get_widget(self):
        return self.widget


    # called by the dashboard when a new chat log is uploaded
    # the new user_info is passed, and the skeleton can be updated to reflect it.
    def init_widget_data(self, user_info):
        self.widget.description = user_info["user_name"]

