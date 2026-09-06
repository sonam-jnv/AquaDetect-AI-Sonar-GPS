# AquaDetect-AI-Sonar-GPS

### Project Description
**AquaDetect-AI-Sonar-GPS** is an autonomous underwater marine debris detection, classification, and geo-telemetry tracking system powered by **Ultralytics YOLOv8** and simulated **4-Satellite Marine GPS telemetry**. It processes Forward-Looking Sonar (FLS) acoustic frames to localize submerged debris and logs spatial coordinates with depth and compass heading on an interactive geospatial map.

---

### 📊 Dataset Details
- **Total Images**: 1,868 Forward-Looking Sonar (FLS) acoustic images
- **Dataset Split (80 / 10 / 10)**:
  - **Train Set**: 1,494 images
  - **Validation Set**: 186 images
  - **Test Set**: 188 images
- **11 Classes & Object Distribution**:
  - `bottle`: 468
  - `tire`: 615
  - `can`: 355
  - `drink-carton`: 348
  - `chain`: 320
  - `valve`: 239
  - `propeller`: 197
  - `hook`: 171
  - `shampoo-bottle`: 110
  - `standing-bottle`: 65
  - `wall`: 994

---

### 🚀 Key Features
1. **Dual-Model YOLOv8 Acoustic Object & Wildlife Detection**:
   - Detects and localizes submerged marine debris using fine-tuned weights (`models/best.pt`).
   - Detects and monitors protected marine wildlife (Dolphins, Sea Turtles, Fish, Sharks, etc.) using `yolov8n.pt`.
2. **Protected Marine Life Protocol & Lime Green Boxes**:
   - Animals are enclosed in **Lime Green (#00FF00) bounding boxes** labeled:
     `"MARINE LIFE: {class} {conf} - PROTECTED"`
   - Map Popup Protocol: `"PROTECTED MARINE LIFE - No cleanup action, monitor only"`.
3. **Priority Eco-Alert Warning**:
   - If both marine debris and protected wildlife are detected in the same frame, the system triggers:
     `"⚠️ Debris near marine life - priority eco-alert"`
4. **Unknown Debris & Hazard Verification (Confidence < 0.40)**:
   - If debris confidence is below 0.40 or no standard object is recognized, the system triggers a **Yellow Warning Bounding Box** labeled:
     `"UNKNOWN DEBRIS / POTENTIAL HAZARD - Needs Verification"`
   - Drops a distinct **Yellow Hazard Pin** on the map.
5. **4-Satellite GPS Geo-Tagging & Folium Telemetry Map**:
   - Computes realistic marine coordinates (`lat`, `lon`, `depth`, `direction`) with 4-satellite lock metadata (`SAT_GPS_12, SAT_GPS_18, SAT_GLONASS_05, SAT_GALILEO_24`).
   - Interactive Folium map with color-coded markers:
     - 🟢 **Marine Animal Protected**: Green
     - 🔵 **Bottle**: Blue
     - 🔴 **Can**: Red
     - ⚫ **Tire**: Black
     - 🟠 **Chain**: Orange
     - 🟡 **Unknown / Hazard**: Yellow
   - Interactive popups displaying debris/wildlife type, confidence score, coordinates, seabed depth (m), and direction.
6. **GPS Historical Logging & CSV Export**:
   - Automatically logs all survey detections to `detections_log.csv` with `is_protected` boolean flag, eco-alert tracking, object counts, and CSV download button.

---

### 🛠️ How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Dataset & Generate GPS Data** *(Optional / Already Generated)*:
   ```bash
   python data_preprocess.py
   ```

3. **Train / Fine-tune YOLOv8** *(Optional / Pre-trained weights available in `models/best.pt`)*:
   ```bash
   python train_yolo.py
   ```

4. **Launch Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```
