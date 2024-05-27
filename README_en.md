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
# the installation folder is target/dw514/install/usr/local/driveworks/samples
```

### build cgf schedule

the schedule yaml, stm, html are auto-generated in `target/dw514/schedule/cgf_custom_nodes`, which is defined in `driveworks-5.14/samples/src/cgf_nodes/CMakeLists.txt`

```sh
target/dw514/schedule/cgf_custom_nodes/DWCGFHelloworld__standardSchedule.stm
target/dw514/schedule/cgf_custom_nodes/DWCGFHelloworld__standardSchedule.yaml
```

### release to orin

cp the install folder `target/dw514/install/usr/local/driveworks/samples` to drive-orin-machine

the recommended way is to use rsync, first of all install the rsync in orin.

```sh
sudo apt install rsync
```

```sh
rsync -avzhlce ssh ./target/dw514/install nvidia@192.168.137.113:~/zhensheng/orin_ws/nv_driveworks/target/dw514/ \
-Pi \
--update \
--inplace  --delete-delay --compress-level=3  --safe-links --munge-links \
--max-delete=5000
```

### run cgf app

```sh
# in orin
cd ~/zhensheng/orin_ws/nv_driveworks/target/dw514/usr/local/driveworks/samples
sudo ./bin/cgf_custom_nodes/example/runHelloworld.sh
# config the ip stack if it prompts
# check the logFolder log, remove the log before next running app
sudo rm -rf ./LogFolder/
```

### modify the graph and run again (nvsci bug)

