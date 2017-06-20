# Overview
python-multi-array exports ```std::shared_ptr<boost::multi_array<T, N>>``` for 
```
T: bool, uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32 and floa64
N: 1 to 8.
```

The library serves a powerful tool to cooperate python and its extension modules written in C++.
You can create arrays and set values in python, make heavy calculation in C++ (OpenMP and every other C++ tools are available,) and get back to python and display the results using matplotlib.

The array is allocated via multi_array.make.
```python
>>> import numpy, multi_array
>>> multi_array.make((4, 2), numpy.float32)
<multi_array.shared_float_matrix object at 0x10aeb3f50>)
```

The array itself has simple I/O APIs:
```python
>>> x = multi_array.make((4, 2), numpy.float32)
>>> x.num_dimensions()
2
>>> x.num_elements()
8
>>> x.shape()
(4, 2)
>>> x[1, 1] = 3
>>> x[1, 1]
3.0
```

and conversion APIs with numpy.ndarray.
```python
>>> x.set(scipy.rand(4, 2))
>>> type(x.get())
<type 'numpy.ndarray'>
>>> x.get()
array([[ 0.91382688,  0.374331  ],
       [ 0.43389955,  0.5571261 ],
       [ 0.6937117 ,  0.40599877],
       [ 0.80906659,  0.75029951]], dtype=float32)
```

# Install

```
$ make
```

You may change the include directory by editing setup.py.

# Requirements

boost 1.63.0+
