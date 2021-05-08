import ipywidgets as widgets

# widgets for the dashboard follow a similar pattern, as follows
# note the get_widget and init_widget_data methods.
class ExampleWidget():

    # you can setup the main skeleton of the widget here as instance variables, such 
    # that its elements can be updated when new data comes in.
    # however, it should not rely on the user_info yet, because that is not available when the app first starts.
    def __init__(self):
        self.widget = widgets.Text(description="")


    # returns the widget skeleton. this is used when the dashboard is first displayed since 
    # Voila cannot re-display new elements after its initial loading in the browser.
    def get_widget(self):
        return self.widget


    # called by the dashboard when a new chat log is uploaded
    # the new user_info is passed, and the skeleton can be updated to reflect it.
    def init_widget_data(self, user_info):
        self.widget.description = user_info["user_name"]

