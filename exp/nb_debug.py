
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/debug_lib.ipynb

#----------------------------------------------------
# Use Pytorchviz and make_dot library to generate model's back propagation
# graph starting from the loss node
#----------------------------------------------------
def show_graph(model, loss, file_path=None):
  !pip install torchviz
  from torchviz import make_dot

  graph = make_dot(loss, params=dict(model.named_parameters()))
  if (file_path is not None):
    graph.render(file_path, format="png")
  return (graph)

#----------------------------------------------------
# Debug Graph Callback to generate dynamic computation graph
#----------------------------------------------------
class DebugGraphCB(Callback):
  def end_tr(self, ctx):
    ctx.graph = show_graph(ctx.model, ctx.loss)