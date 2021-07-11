import mandatory_libraries as ml

def generate_datatable(dataframe):
    dataframe_df = dataframe
    css = ['https://cdn.datatables.net/1.10.24/css/jquery.dataTables.min.css',
           ]
    js = {
        '$': 'https://code.jquery.com/jquery-3.5.1.js',
        'DataTable': 'https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js',
    }
    ml.pn.extension(css_files=css, js_files=js)

    script = """
    <script>
    if (document.readyState === "complete") {
      $('."""+'front_end_title'+"""').DataTable();
    } else {
      $(document).ready(function () {
        $('."""+'front_end_title'+"""'').DataTable();
      })
    }
    </script>
    """

    html = dataframe_df.to_html(classes=['front-end_title', 'dataframe_df'])
    return ml.pn.pane.HTML(html+script, sizing_mode='stretch_width')

