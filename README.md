# PolarSearch: a vector similarity searcher

## Tips

- reprozip

```
# Create Exp ENV
reprozip trace ./myprogram

# Create Pack
reprozip pack myprogram.rpz

# Run Pack
reprozip unpack myprogram.rpz
reprozip docker setup
reprozip docker run

```

- dummy test

```
cd scripts
make
./data_gen 5000000 10
```

## TODO
