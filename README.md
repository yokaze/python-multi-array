# python-multi-array
A python wrapper for boost::multi_array

# Install

```
$ make
```

You may change the include directory by editing setup.py.

# Usage

```python
>>> import multi_array, scipy
>>> x = multi_array.make((2, 4), multi_array.float32)
>>> x.set(scipy.rand(2, 4))
>>> x.get()
array([[ 0.9504686 ,  0.37968314,  0.93876582,  0.17501496],
       [ 0.87398338,  0.83764452,  0.69423091,  0.29122856]], dtype=float32)
>>> x[1, 2]
0.6942309141159058
```
