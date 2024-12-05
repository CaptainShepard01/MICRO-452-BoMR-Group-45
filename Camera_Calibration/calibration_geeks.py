# Import required modules 
import cv2 
import numpy as np
import glob 


# Define the dimensions of checkerboard 
CHECKERBOARD = (9, 7) 
square_size  = 20e-3


# stop the iteration when specified 
# accuracy, epsilon, is reached or 
# specified number of iterations are completed. 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 


# Vector for 3D points 
threedpoints = [] 

# Vector for 2D points 
twodpoints = [] 


# 3D points real world coordinates 
objectp3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objectp3d *= square_size
prev_img_shape = None


# Extracting path of individual image stored 
# in a given directory. Since no path is 
# specified, it will take current directory 
# jpg files alone 
images = glob.glob('images_calibration/Calib5/*.jpg') 
print(images)

for filename in images:
    image = cv2.imread(filename) 
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Find the chess board corners 
    # If desired number of corners are 
    # found in the image then ret = true 
    ret, corners = cv2.findChessboardCorners( 
                    grayColor, CHECKERBOARD, 
                    cv2.CALIB_CB_ADAPTIVE_THRESH 
                    + cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE) 

    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret == True: 
        threedpoints.append(objectp3d) 

        # Refining pixel coordinates 
        # for given 2d points. 
        corners2 = cv2.cornerSubPix( 
            grayColor, corners, (11, 11), (-1, -1), criteria) 

        twodpoints.append(corners2) 

        # Draw and display the corners 
        image = cv2.drawChessboardCorners(image, 
                                        CHECKERBOARD, 
                                        corners2, ret) 

    cv2.imshow('img', cv2.resize(image, (960, 540))) 
    cv2.waitKey(0) 

cv2.destroyAllWindows() 

h, w = image.shape[:2] 


# Perform camera calibration by 
# passing the value of above found out 3D points (threedpoints) 
# and its corresponding pixel coordinates of the 
# detected corners (twodpoints) 
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
    threedpoints, twodpoints, grayColor.shape[::-1], None, None) 


# Displaying required output 
print(" Camera matrix:") 
print(matrix) 

print("\n Distortion coefficient:") 
print(distortion) 

print("\n Rotation Vectors:") 
print(r_vecs) 

print("\n Translation Vectors:") 
print(t_vecs) 




# Camera calibration parameters from the previous calibration
# Replace these with the actual values you obtained
camera_matrix = matrix
dist_coeffs = distortion

# Open the webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Retrieve the resolution of the video feed
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from video capture")
    cap.release()
    exit()

h, w = frame.shape[:2]

# Compute the optimal new camera matrix
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# Start video feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video capture")
        break

    # Undistort the image
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Display the original and corrected video feeds
    cv2.imshow("Original Video", frame)
    cv2.imshow("Corrected Video", undistorted_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
