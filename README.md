# PolarSearch: a vector similarity searcher

## Tips

- reprozip

```
# Create Exp ENV
reprozip trace ./data_gen

# Create Pack
reprozip pack data_gen.rpz

# Run Pack
reprounzip docker setup data_gen.rpz data_gen.docker
reprozip docker setup
reprozip docker run

```

- dummy test

```
cd scripts
make
./data_gen 5000000 10
```

- Cache Line of CPU

```
root@iZbp1ajubcf4dil5d8ma6gZ:~# lscpu | grep "cache"
L1d cache:                          512 KiB (16 instances)
L1i cache:                          512 KiB (16 instances)
L2 cache:                           16 MiB (16 instances)
L3 cache:                           33 MiB (1 instance)
```

## TODO
