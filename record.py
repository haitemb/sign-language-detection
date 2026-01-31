import cv2
import time
import os
import shutil

label = input("Class name: ").strip()
samples = int(input("how many videos to record: "))

save_path = f"data/{label}"

if os.path.exists(save_path):
    choice = input("Folder exists. Add more videos? (y/n): ").strip().lower()
    if choice == 'y':
        # continue numbering
        existing_videos = [f for f in os.listdir(save_path) if f.endswith(".mp4")]
        if existing_videos:
            # find last number
            nums = []
            for f in existing_videos:
                try:
                    nums.append(int(f.split("_")[-1].split(".")[0]))
                except:
                    pass
            start_count = max(nums)
        else:
            start_count = 0
    else:
        # delete old and start fresh
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        start_count = 0
else:
    os.makedirs(save_path, exist_ok=True)
    start_count = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

print("\nCamera ready")
print("Press ENTER to record video | press Q to quit\n")

count = start_count
while count < start_count + samples:
    ret, frame = cap.read()
    cv2.imshow("preview", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == 13:  # ENTER
        count += 1
        print(f"video {count}/{start_count + samples} will start in 2 seconds...")
        time.sleep(2)

        filename = f"{save_path}/{label}_{count}.mp4"
        out = cv2.VideoWriter(filename, fourcc, 30, (256,256))

        record_frames = 90   # 3 seconds @30fps

        for i in range(record_frames):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (256,256))
            out.write(frame)
            cv2.imshow("preview", frame)
            cv2.waitKey(1)

        out.release()
        print("saved:", filename)

cap.release()
cv2.destroyAllWindows()
