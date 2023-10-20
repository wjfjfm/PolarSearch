
# python NGT

[日本語](README-jp.md)

## Install
Python binding with pybind11 (ngtpy) is installed as follows.
```
pip3 install ngt
```
You can install the python bindings from source code. You **MUST** install the NGT library according to the [README](../README.md#build) before installing the python bindings as follows.
```
pip3 install pybind11
pip3 install numpy
cd NGT_ROOT/python
python3 setup.py sdist
pip3 install dist/ngt-x.x.x.tar.gz
```
Please note that the search speed of the ngtpy packages from PyPI is slower than that of the ngtpy that is built on your computer so that the package can be run on older CPUs.  

## Documents

[ngtpy (pybind11) reference](README-ngtpy.md)

## Simple samples

### ngtpy (pybind11)

ngtpy(pybind11) can reduce the processing times than ngt(ctypes). It is more effective especially for the short search time. 

```python
  import ngtpy
  import random


  dim = 10
  nb = 100
  vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
  query = vectors[0]
  
  ngtpy.create(b"tmp", dim)
  index = ngtpy.Index(b"tmp")
  index.batch_insert(vectors)
  index.save()

  results = index.search(query, 3)
  for i, (id, distance) in enumerate(results) :
      print(str(i) + ": " + str(id) + ", " + str(distance))
      object = index.get_object(id)
      print(object)

```

See also [sample.py](sample/sample.py).

