import numpy as np
import cv2 

class Sift():
    def __init__(self, image_path):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #query image
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img, None)

        #Feature matching
        self.index_params = dict(algorithm=0, trees=5)
        self.search_params = dict()
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
    
    def run(self, frame, return_homography):
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #train image

        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)
        # grayframe = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)
        matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_points.append(m)

        if return_homography == False:
            img3 = cv2.drawMatches(self.img, self.kp_image, grayframe, kp_grayframe, good_points, grayframe)
            return img3

        # Homography    
        if len(good_points) > 10:
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1,2)

            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # perspective transform
            h, w = self.img.shape
            pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            return homography*3
        
        else:
            return grayframe

cap = cv2.VideoCapture(0)
robot_detect = Sift("robot_guy.jpg")
while (True):
    _, frame = cap.read()
    cv2.imshow('robot detect', robot_detect.run(frame, True))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()