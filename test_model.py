from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
#works for Bacterial leaf blight, Brown spot, leaf, leaf_smut
def load_image(img_path):
	# load the image
	img = load_img(img_path, target_size=(150, 150))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 150, 150, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# Function to show array of images (intermediate results)
# Function to show array of images (intermediate results)
def show_images(images):
    for i, img in enumerate(images):
        width ,height = 1000 ,500
        img = cv2.resize(img,(width,height))
        #img = cv2.resize(img,None,fx=0.5,fy=0.5)
        cv2.imshow("image_" + str(i), img)
    cv2.waitKey()
    #cv2.destroyAllWindows()

img_path = "final_data/test/Bacterial leaf blight/DSC_0380.JPG"

# Read image and preprocess
image = cv2.imread(img_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)

edged = cv2.Canny(blur, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#show_images([blur, edged])

# Find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts, _) = contours.sort_contours(cnts)

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 100]

#cv2.drawContours(image, cnts, -1, (0,255,0), 3)

#show_images([image, edged])
#print(len(cnts))

# Reference object dimensions
# Here for reference I have used a 2cm x 2cm square
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2
pixel_per_cm = dist_in_pixel/dist_in_cm

# Draw remaining contours
for cnt in cnts:
	box = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(tl, tr, br, bl) = box
	cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
	mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
	mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
	wid = euclidean(tl, tr)/pixel_per_cm
	ht = euclidean(tr, br)/pixel_per_cm
	cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
		cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
	cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
		cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
def run_example():
    # load the image
    img = load_image(img_path)
    # load model
    model = load_model('final_model1.h5')
    # predict the class
    result = model.predict_classes(img)
    if result == 0:
        print('diseased coffee leaves')
        print('disease name is health')
        print('class number is:')
        print(str(result[0]))
    if result == 1:
        print('diseased coffee leaves')
        print('disease name is miner')
        print('class number is:')
        print(str(result[0]))
    if result == 2:
        print('diseased coffee leaves')
        print('disease name is rust')
        print('class number is:')
        print(str(result[0]))
    if result == 3:
        print('diseased coffee leaves')
        print('disease name is phoma')
        print('class number is:')
        print(str(result[0]))
    if result == 4:
        print('diseased coffee leaves')
        print('disease name is cercospora')
        print('class number is:')
        print(str(result[0]))
    if result == 5:
        print('diseased rice leaves')
        print('disease name is bacterial leaf blight')
        print('class number is:')
        print(str(result[0]))
    if result == 6:
        print('diseased rice leaves')
        print('disease name is brown spot')
        print('class number is:')
        print(str(result[0]))
    if result == 7:
        print('diseased rice leaves')
        print('disease name is leaf smut')
        print('class number is:')
        print(str(result[0]))
    if result == 8:
        print('healthy leaves')
        print('class number is:')
        print(str(result[0]))
show_images([image])
run_example()
