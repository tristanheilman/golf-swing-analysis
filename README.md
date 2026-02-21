# Golf Swing Analysis using MediaPipe

![Banner](assets/banner.png)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green.svg)](https://opencv.org/)
[![Mediapipe](https://img.shields.io/badge/Mediapipe-Latest-orange.svg)](https://mediapipe.dev/)
![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4-lightpink)
[![By HeleenaRobert](https://img.shields.io/badge/By-HeleenaRobert-purple?logo=github)](https://github.com/HeleenaRobert)

A Python project that analyzes golf swings using **MediaPipe Pose**. It tracks body landmarks (ear, hip, wrist) and the golf ball, calculates swing angles, overlays X/Y axes for reference, and saves an annotated swing video.

---

## ✨ Features

- Processes golf swing videos frame-by-frame.
- Detects **ear, hip, wrist** positions using MediaPipe Pose.
- Draws swing lines and wrist-to-ball connection.
- Calculates and overlays **swing angle (ear-hip-wrist)** per frame.
- Adds an **X/Y axis overlay** for spatial reference.
- Saves annotated video automatically in `output/`.

---

## 📂 Folder Structure

```Structure
golf-swing-analysis/
│
├── golf_swing_analysis.py
│
├── utils/
│   ├── swing_utils.py
│   └── video_utils.py
│
├── input/
│   └── Reference_Swing_DTL.mp4
│
├── output/
│   └── Reference_Swing_DTL_out_landmarks.mp4
│
├── assets/
│   ├── banner.png
│   └── video_sample_image.png
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## 🚀 How It Works

1. Reads an input swing video from `input/`.
2. Processes each frame with **MediaPipe Pose**.
3. Draws landmarks, swing lines, angle text, and axes.
4. Writes the annotated frames into an output video.

---

## 🖼 Sample Output

_(Output video will be saved in `output/` as MP4.)_

 ![Example_image](assets/video_sample_image.png)

---

## 🔧 Installation

```bash
git clone https://github.com/HeleenaRobert/golf-swing-analysis.git
cd golf-swing-analysis

# Windows
pip install -r requirements.txt

# Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Place your swing video in the `input/` folder.
2. Run the script:

   ```bash
   python golf_swing_analysis.py
   ```

3. Processed video will be saved in `output/`.

---

## 📌 Notes

- Ball position is fixed (adjust `BALL_POS` in `swing_utils.py` for different videos).
- Angle calculation is currently **ear–hip–wrist**; more angles can be added.
- Works best with **down-the-line swing videos**.

---

## 🛠 Technologies Used

- [Python 3.8+](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [MediaPipe Pose](https://mediapipe.dev/solutions/pose.html)
- [NumPy](https://numpy.org/)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👩‍💻 Author

**Heleena Robert**  
[GitHub](https://github.com/HeleenaRobert)
