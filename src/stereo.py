"""
Stereo image matching and point cloud generation
Licence: BSD
Author : Hoang Anh Nguyen
"""

import cv2
import numpy as np
import argparse, sys

#--------------------------------------------------------
#--------------------------------------------------------
# Class provide an interface to perform stereo matching
class StereoMatching:
    
    def __init__(self, limage_name, rimage_name):
        # read image files and convert it to grayscale
        self.left_image      = cv2.imread(limage_name)
        self.left_gray_image = cv2.cvtColor(self.left_image,
                                            cv2.COLOR_BGR2GRAY)
        
        self.right_image      = cv2.imread(rimage_name)
        self.right_gray_image = cv2.cvtColor(self.right_image,
                                             cv2.COLOR_BGR2GRAY)
        
    #--------------------------------------------------------
    # Main methods
    #--------------------------------------------------------
    
    # Extract text locations in the image using a ER method
    # input:    method ID
    # return:   point cloud based on disparity map
    def matching(self, methodId = 1):
        methodName = [
            'StereoBM',
            'StereoSGBM'
        ][methodId - 1]
        self.stereo = eval(methodName)()
        
        # compute disparity map
        self.disp = self.stereo.generateDisparity(self.left_gray_image,
                                                  self.right_gray_image)
        
        # Generate point cloud based on disparity map
        return self.generate_point_cloud()
        
    def generate_point_cloud(self):
        # point cloud
        h, w = self.left_image.shape[:2]
        f = 0.8 * w             # guess for focal length
        Q = np.float32([[1,  0, 0, -0.5 * w],
                        [0, -1, 0,  0.5 * h], # turn points 180 deg around x-axis,
                        [0,  0, 0,       -f], # so that y-axis looks up
                        [0,  0, 1,        0]])
        points = cv2.reprojectImageTo3D(self.disp, Q)
        colors = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2RGB)
        mask = self.disp > self.disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        out_fn = 'output.ply'
        
        # write to PLY file
        StereoMatching.write_ply('output.ply', out_points, out_colors)
        print('%s saved' % 'out.ply')
        
        min_disp = 16#self.stereo.getMinDisparity()
        num_disp = 112#self.stereo.getNumDisparities()
        return (self.disp - min_disp) / num_disp
            
    @staticmethod
    def write_ply(fn, verts, colors):
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
        '''
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

#--------------------------------------------------------
#--------------------------------------------------------
class Stereo:
    
    def __init__(self):
        self.stereo = None
        
    def generateDisparity(self, left_image, right_image):
        if not self.stereo:
            return None
        return self.stereo.compute(left_image,
                                   right_image).astype(np.float32) / 16.0

# Block matching algorithm
# http://docs.opencv.org/master/d9/dba/classcv_1_1StereoBM.html
class StereoBM(Stereo):
    
    def __init__(self):
        self.stereo = cv2.StereoBM_create()
       
# Class modified H. Hirschmuller algorithm
# http://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html
class StereoSGBM(Stereo):
    
    def __init__(self):
        window_size = 3
        min_disp = 16
        num_disp = 112 - min_disp
        # using modified H. Hirschmuller algorithm
        self.stereo = cv2.StereoSGBM_create(
            minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 16,
            P1 = 8 * 3 * window_size**2,
            P2 = 32 * 3 * window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )
        
#--------------------------------------------------------
#--------------------------------------------------------
def main(argv):
    # Define argument list. Example:
    # python stereo.py  -m 1 \
    #                   -l test/imL2.bmp \
    #                   -r test/imL2l.bmp \
    #                   -o .
    parser = argparse.ArgumentParser(description='Scene Text Detection and Recognition')
    parser.add_argument('-m','--method',
                        help="""Stereo matching method:
                        1: BM
                        2: SGBM
                        """,
                        required=True)
    parser.add_argument('-l','--left',
                        help='Left image',
                        required=True)
    parser.add_argument('-r','--right',
                        help='Right image',
                        required=True)
    parser.add_argument('-o','--output',
                        help='Ouput location',
                        required=True)
    args = vars(parser.parse_args())
    
    # extract arguments
    methodId = int(args['method'])
    
    # stereo matching
    stereo = StereoMatching(args['left'], args['right'])
    output = stereo.matching(methodId)
    cv2.imwrite(args['output'] + '/output.jpg', output)
    print("Output saved!")
    
if __name__ == '__main__':
    main(sys.argv)
