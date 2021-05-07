import ipywidgets as widgets

def get_basic_info_widget(user_info):
    html = """\
    <p><b>Username:</b> {}<p>
    <p><b>Number of Messages:</b> {}<p>
    <p><b>Message Samples:</b> <br>{}<p>

    """
    html = html.format(user_info["user_name"], len(user_info["user_messages"]), "<br>".join(user_info["user_messages"][:10]))
    widget = widgets.HTML(html)
    return widget