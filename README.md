# Stereo image matching and point cloud generation

This project aims to perform stereo matching and generate point cloud based on the disparity map.

![Matching result](/demo.jpg)

## Installation OpenCV on Mac OS
This is summarized from http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/

1. Requires Python 2.7, Git
2. Install Python packages
  ```
pip install -r requirement.txt
  ```
  
3. Install necesssary packages
  ```
brew install cmake pkg-config jpeg libpng libtiff openexr eigen tbb
  ```
  
4. Clone OpenCV repos
  * opencv
 ```
cd ~
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout master
 ```
   * opencv_contrib
   ```
cd ~
git clone https://github.com/Itseez/opencv_contrib
cd opencv_contrib
git checkout master
   ```
   
5. Build and install OpenCV
  ```
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
	-D PYTHON2_PACKAGES_PATH=~/feature/lib/python2.7/site-packages \
	-D PYTHON2_LIBRARY=/usr/bin/python \
	-D PYTHON2_INCLUDE_DIR=/System/Library/Frameworks/Python.framework/Headers \
	-D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
	-D BUILD_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules ..
make -j4
make install
  ```
  
6. Verify in python shell
  ```python
>>> import cv2
>>> cv2.__version__
'3.1.0-dev'
>>> help(cv2.text)
  ```

## Usage

1. To show all available options:
  ```
  python stereo.py -h
  ```

  An example:

  ```
  python stereo.py -m <1: BM, 2: SGBM> \
		           -l <path to left image> \
                   -r <path to right image> \
  		           -o <path to output image>
  ```

2. The point cloud is generated to PLY format, which can be visualized using [Meshlab](http://meshlab.sourceforge.net/). An example of Meshlab:

  ![Matching result](/demo.jpg)