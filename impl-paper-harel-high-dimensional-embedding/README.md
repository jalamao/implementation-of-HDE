# About

The paper _Graph Drawing by High-Dimensional Embedding_[^1] presents a approach to draw undirected graphs. This is a Python implementation of the apporach.

# How to use

```python
from algorithm import GraphDrawing

gd = GraphDrawing() # default dimension is 50
gd.transform('data/grid_50x50.csv')

# now you can retrieve tranformed points by accessing `gd.transformed_points`.

gd.plot("figure.png") # drawing image is slow when solving for large number (1000's) of points
```

# Examples

The following are example drawings. Original images are placed in the folder [images](./images).

![](images/demo.png)


[^1]: http://emis.ams.org/journals/JGAA/accepted/2004/HarelKoren2004.8.2.pdf
