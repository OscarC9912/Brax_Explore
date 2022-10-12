"""
Do some tree operations    
"""
import inspect

def tree_flatten(obj):
    pytree_data = []
    pytree_fields = []
    static_data = {}
    static_fields = set()
    
    # commented part deals with attributes that should be ignored
    for c in inspect.getmro(obj.__class__):
      if hasattr(c, '__pytree_ignore__'):
        static_fields.update(obj.__class__.__pytree_ignore__)
    
    for k, v in vars(obj).items():
      if k in static_fields:
        static_data[k] = v
      else:
        pytree_fields.append(k)
        pytree_data.append(v)
    return (pytree_data, (pytree_fields, static_data))
  
  
  