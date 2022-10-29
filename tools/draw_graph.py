import graphviz

graph = graphviz.Digraph('image_flow', filename='graph',format='png')

graph.attr('node', shape='diamond', style='filled', color='#DDFFAA')
graph.node('net1', label='net')
graph.node('net2', label='net')

graph.attr('node', shape='box', style='', color='#000000')
graph.node('img_var', label='Img_var\n(1,256,256)')
graph.node('img_noise_var', label='Img_noise_var\n(1,256,256)')
graph.node('img_clean_var', label='img_clean_var\n(1,256,256)')

with graph.subgraph(name='cluster') as subgraph:
    subgraph.attr(style='filled', color='lightgrey')
    subgraph.node_attr.update(style='filled', color='white')

    subgraph.edges([
        ('Net_input','Net_input_saved'),
        ('Net_input', 'noise'),
        ('Net_input_saved', 'Net_input_new'),
        ('Net_input_new', 'net1'),
        ('Net_input_saved', 'net2'),
        ('net1', 'out1'),
        ('net2', 'out2'),
    ])
    subgraph.attr(label='ineration')

    subgraph.node('Net_input', label='Net_input\n(1,1,256,256)\n# zeros')
    subgraph.node('Net_input_saved', label='Net_input_saved\n(1,1,256,256)\n# clone')
    subgraph.node('noise', label='noise\n(1,1,256,256)\n# clone')

    subgraph.node('Net_input_new', label='Net_input\n(1,1,256,256)\n# 94 input+noise')

    subgraph.node('out1', label='out1\n(1,1,256,256)\n# out1')
    subgraph.node('out2', label='out2\n(1,1,256,256)\n# out2')


graph.edge('img_var', 'img_noise_var')
graph.edge('img_var', 'img_clean_var')
graph.edge('img_clean_var', 'Net_input')
# graph.edge('Net_input', 'Net_input_saved')
# graph.edge('Net_input', 'noise')
# graph.edge('Net_input_saved', 'Net_input_new')
# graph.edge('Net_input_new', 'net1')
# graph.edge('Net_input_saved', 'net2')
# graph.edge('net1', 'out1')
# graph.edge('net2', 'out2')

# graph.edges([
#     ('Net_input_saved','noise'),
#     ('out1','img_noise_var'),
#     ('out2','img_clean_var'),
# ])



graph.render('graph', view=True)
