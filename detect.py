import cv2
from ultralytics import YOLO

def main():
    # Get IP webcam URL from the user
    ip_webcam_url = input("Enter the IP webcam URL (e.g., http://<IP_ADDRESS>:<PORT>/video): ")

    # Path to the YOLO model (choose one of the methods below)
    model_path = r"C:\Users\SRIKANTH\PycharmProjects\yolov8\runs\detect\train2\weights\last.pt"

    # Load YOLO model
    model = YOLO(model_path)

    # Open a connection to the IP webcam
    cap = cv2.VideoCapture(ip_webcam_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    else:
        print("Successfully opened video stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Perform YOLO detection on the frame
        results = model(frame, save=False)  # save=False to not save images

        # Plot the results on the frame
        res_plotted = results[0].plot()

        # Display the frame with detections
        cv2.imshow('YOLO Detection', res_plotted)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
