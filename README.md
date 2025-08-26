# GateGuard AI 🛡️🎥

**GateGuard AI** is an AI-powered real-time people counting system built using Python, OpenCV, and YOLO. It detects people from a video feed and counts how many individuals enter and exit a monitored area by tracking movement across two virtual lines.

---

## 🚀 Features

- 🎯 Real-time human detection using YOLO
- 🧠 Person tracking using unique Track IDs
- 🔴 Entry and 🟣 Exit line detection
- 📊 Live counter showing:
  - Total Entries
  - Total Exits
  - People currently inside
- 🔍 Mouse hover to get pixel coordinates (for custom line placement)
- 💡 Designed for CCTV-style surveillance or access monitoring

---

## 🛠️ Tech Stack

- Python 3.x
- OpenCV
- YOLOv8 (Ultralytics)
- cvzone
- NumPy

---

## 📁 Directory Structure

```
GateGuard-AI/
├── main.py # Main project code
├── yolo11s.pt # YOLO model weights
├── requirements.txt # Required Python packages
└── README.md # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/axixatechnologies/GateGuard-AI.git
   cd GateGuard-AI
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO model**

   - Download your YOLO `.pt` file and place it in the project folder.
   - Example: `yolo11s.pt`

4. **Run the project**

   ```bash
   python main.py
   ```

---

## 🎯 Customization

- Modify line coordinates inside `main.py` to fit your camera view:

  ```python
  line_red = ((x1, y1), (x2, y2))    # Entry line
  line_pink = ((x1, y1), (x2, y2))   # Exit line
  ```

- Use the mouse hover feature to get live pixel positions.

---

## 📸 Demo

Coming soon...

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Developed By

Axixa Technologies – AI Training Team ([Nidhi](https://github.com/nidhiach) And [Khushi](https://github.com/Khushi6367))
