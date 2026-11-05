from detector.model_detector import GestureDetectorConv3d
from models.model_v4 import GestureRecognizer

import cv2
import numpy as np
from collections import deque
import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import time


def resample_list(data, n):
    idxs = np.linspace(0, len(data) - 1, n).round().astype(int)
    return [data[i] for i in idxs]


def get_gesture(clip: torch.Tensor, model: torch.nn.Module) -> tuple[int, float]:
    model.eval()
    with torch.no_grad():
        gesture_prob = model(clip)  # (1, num_classes)
        gesture_prob = torch.softmax(gesture_prob, dim=1)
        gesture_idx = torch.argmax(gesture_prob, dim=1).item()
        gesture_conf = gesture_prob[0, gesture_idx].item()
        return gesture_idx, gesture_conf
    

def preprocess_clip(buffer: deque, resample: int, transform) -> torch.Tensor:
    clip = list(buffer)
    if len(clip) != resample:
        clip = resample_list(clip, resample)
    tensor_clip = torch.stack([transform(img) for img in clip])  # (1, T, C, H, W)
    tensor_clip = tensor_clip.unsqueeze(0)  # (T, C, H, W)
    return tensor_clip



def main():
    # Device configuration
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Load the trained detector model
    detector = GestureDetectorConv3d()
    detector.load_state_dict(torch.load("checkpoints/detector.pth", map_location=device))
    detector.to(device)
    detector.eval()

    # Load the trained recognizer model
    recognizer = GestureRecognizer()
    recognizer.load_state_dict(torch.load("checkpoints/recognizer.pth", map_location=device))
    recognizer.to(device)
    recognizer.eval()

    class_file = 'dataset/class-index.csv'
    class_df = pd.read_csv(class_file, sep=';', header=None, names=['class_idx', 'class_name'])
    class_idx_to_name = dict(zip(class_df['class_idx']-1, class_df['class_name']))

    # Initialize video capture and frame buffer
    MAXLEN = 36
    buffer = deque(maxlen=MAXLEN)  # 36 frame long buffer
    skipcounter = 0

    resize = transforms.Resize((112, 112))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 25)

    lastgesture_time = time.time()
    lastgesture_name = None
    do_puttext = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = resize(Image.fromarray(rgb_frame))
        buffer.append(resized_frame)

        if len(buffer) == MAXLEN:
            if skipcounter == 0:
                # Process the clip
                tensor_clip = preprocess_clip(buffer, 20, transform)
                tensor_clip = tensor_clip.to(device)

                detector_idx, detector_conf = get_gesture(tensor_clip, detector)

                if detector_idx == 1 and detector_conf >= 0.7:
                    # Recognize the gesture
                    rec_tensor_clip = preprocess_clip(buffer, MAXLEN, transform)
                    rec_tensor_clip = rec_tensor_clip.to(device)
                    gesture_idx, gesture_conf = get_gesture(rec_tensor_clip, recognizer)

                    if gesture_idx not in [0, 2] and gesture_conf > 0.4:  # Ignore 'No gesture' and 'Doing other things' gestures
                        gesture_name = class_idx_to_name[gesture_idx]

                        if (lastgesture_name != gesture_name) or (time.time() - lastgesture_time > 2.0):
                            # Publish recognized gesture
                            print(f"Detected: {gesture_name} ({detector_conf:.2f} x {gesture_conf:.2f})")

                            # Settings
                            lastgesture_time = time.time()
                            lastgesture_name = gesture_name
                            do_puttext = True
                            skipcounter = MAXLEN  # Skip next MAXLEN frames to avoid duplicates

                        else:
                            skipcounter = 5
                            print(f"Skipped duplicate gesture: {gesture_name}, time since last: {time.time() - lastgesture_time:.2f}s")

                    else:
                        skipcounter = 4
                    
                else:
                    skipcounter = 4

            else:
                skipcounter -= 1

        # Display the results
        if do_puttext and lastgesture_name is not None:
            elapsed_print_time = time.time() - lastgesture_time
            if elapsed_print_time <= 2.0:
                cv2.putText(frame, f"{lastgesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                do_puttext = False

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()