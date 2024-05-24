# build cgf app with nv driveworks sdk

## helloworld nvsci bug

### build cgf nodes

note: add variable definition in `samples/CMakeLists.txt` to force the sample compile out of `/usr/local/driveworks`

```sh
set(DRIVEWORKS_DIR /usr/local/driveworks)
```

```sh
rm -rf ./target/dw514
cmake -B ./target/dw514/ -DCMAKE_TOOLCHAIN_FILE=/usr/local/driveworks/samples/cmake/Toolchain-V5L.cmake -DVIBRANTE_PDK=/drive/drive-linux -S ./driveworks-5.14/samples
make -C ./target/dw514/ -j $(($(nproc)-1)) install
```

### build cgf schedule

### run cgf app

