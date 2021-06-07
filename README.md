[![CodeFactor](https://www.codefactor.io/repository/github/wonder-tree/posecamera-cpp/badge)](https://www.codefactor.io/repository/github/wonder-tree/posecamera-cpp)

# PoseCamera(cpp)

## Realtime Humanpose Estimation in C++

This is a realtime humanpose estimation using libtorch. This project can runs realtime (16fps) on CPU and about 50fps on GPU. 

## Install


> use `trace_model.py` file to trace your pytorch model for libtorch c++

In order to run this demo, you have to follow these steps:

1. Install thor which provide visualization utils.

   ```
   # Download libtorch
   # install thor
   git clone http://github.com/jinfagang/thor
   cd thor
   mkdir build && cd build
   cmake ..
   make -j8
   sudo make install
   ```

   

2. make executable file

   ```
   mkdir build
   cd build
   cmake .. && make -j4
   ```
      

And that is all!  **make sure you have copied libtorch to your home folder**.
