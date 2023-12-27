import cv2
import time
import numpy as np

from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Open the video file
video_path = 'media/project_video.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)
width = int(cap.get(3))
height = int(cap.get(4))
print("Video dimensions: {} x {}".format(width, height))

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

# # Check if the video file opened successfully
# if not cap.isOpened():
#     print("Error: Couldn't open video file.")
#     exit()

# Initial values for contrast, brightness, and gamma correction
contrast = 1.2
brightness = 0.0
gamma = 2

# Loop through the frames of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # # Break the loop if the video has ended
    # if not ret:
    #     break

    # Apply contrast and brightness adjustments
    enhanced_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # Apply gamma correction
    enhanced_frame = np.power(enhanced_frame / 255.0, gamma) * 255.0
    enhanced_frame = np.uint8(enhanced_frame)

    # Convert the enhanced frame to HSV
    hsv_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)

    # Define a lower and upper threshold for white color in HSV
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 25, 255], dtype=np.uint8)

    # Define a lower and upper threshold for yellow color in HSV
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # Create masks for white and yellow colors
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Combine the masks to get the final mask
    final_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # # Find the coordinates of white and yellow pixels in the original frame
    # pixels = cv2.findNonZero(final_mask)

    # # Extract HSV values of white and yellow pixels
    # if pixels is not None:
    #     hsv_values = [hsv_frame[y[0][1], y[0][0]] for y in pixels]
    #     # print("HSV values of white and yellow pixels:", hsv_values)
    # else:
    #     # print("No white or yellow pixels found.")
    #     pass

    # Define region of interest (ROI) vertices
    roi_vertices = np.array([
        [(220, 690), (570, 450), (770, 450), (1160, 690)]
    ], dtype=np.int32)

    # Extract ROI
    mask = np.zeros_like(final_mask)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_image = cv2.bitwise_and(final_mask, mask)

    # Apply Hough line transformation
    lines = cv2.HoughLinesP(roi_image, rho=1, theta=np.pi/180, threshold=8, minLineLength=0.1, maxLineGap=0.1)

    coordinates1 = []
    coordinates2 = []

    # Draw the lines on a copy of the original frame
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if x1 <= 620:
                coordinates1.append([x1, y1])
            elif x2 >= 640:
                coordinates2.append([x2, y2])
            else:
                pass

    # Convert lists to NumPy arrays
    coordinates1 = np.array(coordinates1)
    coordinates2 = np.array(coordinates2)

    # Create a blank image to draw the lines
    line_image_coordinates1 = np.zeros_like(frame)
    # Create a blank image to draw the lines
    line_image_coordinates2 = np.zeros_like(frame)

    # Draw the lines on the blank image
    for coord in coordinates1:
        cv2.line(line_image_coordinates1, (coord[0], coord[1]), (coord[0], coord[1]), (255, 0, 0), 5)
    for coord in coordinates2:
        cv2.line(line_image_coordinates2, (coord[0], coord[1]), (coord[0], coord[1]), (255, 0, 0), 5)



    # Display the image with lines from coordinates1
    #cv2.imshow('Lines from coordinates1', line_image_coordinates1)
    #cv2.imshow('Lines from coordinates2', line_image_coordinates2)

    # Combine the line image with the original frame
    result_image = cv2.addWeighted(line_image, 0.8, line_image, 1, 0)
    # Display the result
    #cv2.imshow('Hough Lines', result_image)
    # Display the original frame, enhanced frame, and the combined mask
    #cv2.imshow('Original Frame', frame)
    #cv2.imshow('Enhanced Frame', enhanced_frame)
    #cv2.imshow('Combined Mask', roi_image)
    # Display the image with lines from coordinates1

    # Check if there are enough points for RANSAC
    if len(coordinates1) < 2:
        print("Left no points")
        
    elif len(coordinates2) < 2:
        print("Right no points")
    else:
        try:
            # Extract x and y coordinates
            x_coords = coordinates1[:, 0].reshape(-1, 1)
            y_coords = coordinates1[:, 1]
            # Create a RANSAC regressor with a polynomial model
            model = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor(LinearRegression()))
            #model = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor(LinearRegression(), max_trials=100, residual_threshold=1.0))
            # Fit the model to the data
            model.fit(x_coords, y_coords)
            # Generate x values for the fitted curve
            x_fit = np.linspace(min(x_coords), max(x_coords), 100).reshape(-1, 1)
            #print(x_fit)
            #x_fit = np.flip(x_fit) 
            # Predict y values using the fitted model
            y_fit = model.predict(x_fit)
            # Draw the fitted curve on the original frame
            for i in range(len(x_fit) - 1):
                cv2.line(frame, (int(x_fit[i]), int(y_fit[i])), (int(x_fit[i + 1]), int(y_fit[i + 1])), (255, 0, 0), 5)
            cv2.putText(frame, f"{(int(x_fit[i]), int(y_fit[i]))}", (int(x_fit[i]), int(y_fit[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  


            # Extract x and y coordinates
            x_coords_2 = coordinates2[:, 0].reshape(-1, 1)
            y_coords_2 = coordinates2[:, 1]
            # Create a RANSAC regressor with a polynomial model
            model_2 = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor(LinearRegression()))
            # Fit the model to the data
            model_2.fit(x_coords_2, y_coords_2)
            # Generate x values for the fitted curve
            x_fit_2 = np.linspace(min(x_coords_2), max(x_coords_2), 100).reshape(-1, 1)
            x_fit_2 = np.flip(x_fit_2)
            # Predict y values using the fitted model
            y_fit_2 = model_2.predict(x_fit_2)

            for i in range(len(x_fit_2) - 1):
                line_start = (int(x_fit_2[i]), int(y_fit_2[i]))
                line_end = (int(x_fit_2[i + 1]), int(y_fit_2[i + 1]))
                cv2.line(frame, line_start, line_end, (255, 0, 0), 5)

            cv2.putText(frame, f"{line_start}", line_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

           

        except ValueError as e:
            print(f"RANSAC failed for coordinates2: {e}")
        except IndexError as ei:
            print(f"Index Error{ei}")


    # Display the original frame with the fitted curve
    cv2.imshow('Original Frame with Fitted Curve', frame)


    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # Update FPS every 1 second
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        start_time = time.time()
        frame_count = 0

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        while cv2.waitKey(1) & 0xFF != ord(' '):
            pass


# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
