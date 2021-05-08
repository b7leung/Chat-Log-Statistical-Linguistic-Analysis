import ipywidgets as widgets

class BasicInfoWidget():

    def __init__(self):
        self.html = """\
        <p><b>Username:</b> {}<p>
        <p><b>Number of Messages:</b> {}<p>
        <p><b>Message Samples:</b> <br>{}<p>

        """
        self.widget = widgets.HTML(self.html.format("", "", ""))


    def get_widget(self):
        return self.widget


    def init_widget_data(self, user_info):
        self.widget.value = self.html.format(user_info["user_name"], len(user_info["user_messages"]), "<br>".join(user_info["user_messages"][:10]))

