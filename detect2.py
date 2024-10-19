import cv2
from ultralytics import YOLO

def main():
    # Get IP webcam URL from the user
    ip_webcam_url = input("Enter the IP webcam URL (e.g., http://<IP_ADDRESS>:<PORT>/video): ")

    # Path to the YOLO model
    model_path = r"C:\Users\SRIKANTH\Downloads\runs-20240716T072728Z-001\runs\detect\train3\weights\last.pt"  # Updated to use the uploaded file

    # Load YOLO model
    print("Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

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
        try:
            print("Performing detection on the frame...")
            results = model(frame, save=False)  # save=False to not save images
            print("Detection performed successfully.")
        except Exception as e:
            print(f"Error during detection: {e}")
            break

        # Check if results contain detections
        if len(results) > 0:
            print(f"Detections found: {len(results)}")
            # Plot the results on the frame
            try:
                res_plotted = results[0].plot()
            except Exception as e:
                print(f"Error plotting results: {e}")
                res_plotted = frame
        else:
            print("No detections found.")
            res_plotted = frame

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
