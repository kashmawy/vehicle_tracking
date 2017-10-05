from features_extract import get_hog_features
import cv2
from scipy.misc import imread, imresize, imsave
import glob
import matplotlib.pyplot as plt

vehicles = glob.glob('./vehicles/*/*.png')
nonvehicles = glob.glob('./non-vehicles/*/*.png')

print("Vehicles length: ", len(vehicles))
print("NonVehicles length: ", len(nonvehicles))

vehicle_image = imread(vehicles[0])
vehicle_image_t = cv2.cvtColor(vehicle_image, cv2.COLOR_RGB2YCrCb)
vehicle_hog, vehicle_hogged = get_hog_features(vehicle_image_t[:,:,0], 9, 5, 2, True)

plt.subplot(1, 3, 1)
plt.imshow(vehicle_image)
plt.subplot(1, 3, 2)
plt.imshow(vehicle_image_t)
plt.subplot(1, 3, 3)
plt.imshow(vehicle_hogged)
plt.plot()
plt.show()

nonvehicle_image = imread(nonvehicles[0])
nonvehicle_image_t = cv2.cvtColor(nonvehicle_image, cv2.COLOR_RGB2YCrCb)
nonvehicle_hog, nonvehicle_hogged = get_hog_features(nonvehicle_image_t[:,:,0], 9, 5, 2, True)

plt.subplot(1, 3, 1)
plt.imshow(nonvehicle_image)
plt.subplot(1, 3, 2)
plt.imshow(nonvehicle_image_t)
plt.subplot(1, 3, 3)
plt.imshow(nonvehicle_hogged)
plt.plot()
plt.show()
