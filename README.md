# 🔥 FlameX7 AirBurger

<div align="center">

**A Gesture-Controlled Touchless Burger Ordering System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Configuration](#-configuration)

</div>

---
<img width="1536" height="1024" alt="FlameX7 AirBurger" src="https://github.com/user-attachments/assets/c15bf5a9-8158-4e1c-ad4c-c9db49c08b66" />

## 📋 Overview

**FlameX7 AirBurger** is an innovative, touchless ordering system that leverages advanced computer vision and hand gesture recognition to revolutionize the quick-service restaurant experience. Users can browse the menu, select items, and complete payments entirely through intuitive hand gestures—eliminating the need for physical contact with touchscreens or keyboards.

### 🎯 Key Highlights

- **100% Touchless Interface**: Control the entire ordering process with hand gestures
- **Real-time Gesture Recognition**: AI-powered hand tracking and gesture detection
- **Voice Feedback**: Audio notifications for user actions and confirmations
- **Digital Receipts**: Automatic receipt generation in CSV format
- **UPI Payment Integration**: Seamless payment gateway with QR code support
- **Modern UI**: Neon-themed interface with smooth animations

---

## ✨ Features

### 🎮 Gesture Controls
- **Point & Hover**: Navigate through menu items
- **Pinch Gesture**: Select items and confirm actions
- **Hand Tracking**: Real-time hand detection and joint tracking
- **Idle Detection**: Automatic screen timeout for user convenience

### 🍔 Menu Management
- **Multiple Categories**: Signature Burgers, Sides & Add-ons, Drinks
- **Dynamic Pricing**: Flexible pricing system with tax calculations
- **Item Images**: Visual burger displays with high-quality images
- **Cart Management**: Add/remove items with quantity control

### 💳 Payment System
- **UPI Integration**: Direct payment through UPI protocol
- **QR Code Generation**: Dynamic QR codes for payment processing
- **Receipt Management**: Automatic receipt saving with itemized details
- **Tax Calculation**: Integrated GST calculation (5% default)

### 🔊 User Experience
- **Voice Assistance**: Text-to-speech feedback for all interactions
- **Real-time Feedback**: Status messages and confirmations
- **Animated UI**: Smooth transitions and visual feedback
- **OLED Display Ready**: Support for small display devices

---

## 🛠 Tech Stack

| Component | Technology | Icon |
|-----------|-----------|------|
| **Language** | Python 3.8+ | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) |
| **Computer Vision** | OpenCV 4.0+ | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) |
| **Hand Detection** | MediaPipe | ![ML](https://img.shields.io/badge/-MediaPipe-FF6F00?logo=tensorflow&logoColor=white) |
| **Speech Synthesis** | pyttsx3 | ![Audio](https://img.shields.io/badge/-pyttsx3-1f77b4) |
| **QR Code** | qrcode | ![QR](https://img.shields.io/badge/-QRCode-000000?logo=qrcode&logoColor=white) |
| **Configuration** | python-dotenv | ![Config](https://img.shields.io/badge/-dotenv-ECD53F?logo=.env&logoColor=white) |
| **Data Processing** | NumPy | ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera device
- 2GB RAM minimum
- Stable internet connection (for payment processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/WafryAhamed/FlameX7.git
cd FlameX7
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```env
UPI_ID=your_upi_id@upi
UPI_NAME=Your Restaurant Name
UPI_CURRENCY=LKR
```

---

## 🚀 Running the Project

### Quick Start
```bash
python main.py
```

### With Custom UPI Configuration
```bash
# Set environment variables before running
# Windows PowerShell
$env:UPI_ID="custom@upi"; $env:UPI_NAME="Custom Name"; python main.py

# Linux/macOS
UPI_ID=custom@upi UPI_NAME="Custom Name" python main.py
```

### Application Launch
Once started, the application will:
1. Initialize the camera and hand detection models
2. Load burger images and menu data
3. Display the main menu interface
4. Wait for gesture inputs

---

## 📁 Project Structure

```
FlameX7/
├── main.py                          # Main application file
├── requirements.txt                 # Python dependencies
├── .env                             # Configuration file (create this)
├── README.md                        # This file
├── assets/
│   └── burgers/                     # Burger images
│       ├── burger1.png
│       ├── burger2.png
│       ├── burger3.png
│       ├── burger4.png
│       ├── burger5.png
│       ├── burger6.jpg
│       └── burger7.png
├── receipt_*.csv                    # Generated receipts
└── backend_order_log.csv            # Order history log
```

---

## ⚙️ Configuration

### Application Settings
Edit the configuration section in `main.py`:

```python
# Display Settings
WIDTH, HEIGHT = 1000, 640           # Main display resolution
OLED_W, OLED_H = 320, 220          # Secondary display resolution

# Gesture Recognition
HOVER_THRESHOLD = 1.3               # Hover detection sensitivity
PINCH_DIST_PX = 35                  # Pinch gesture threshold

# Animation
ANIM_FRAMES = 14                    # Animation frame count
ANIM_FPS_DELAY = 0.016              # Frame delay in seconds

# User Interaction
IDLE_TIMEOUT = 60.0                 # Idle timeout in seconds (default: 60s)
SMOOTH_ALPHA = 0.65                 # Hand tracking smoothing factor
```

### Payment Configuration
Update the UPI settings in `.env`:

```env
UPI_ID=merchant_upi_id@bank
UPI_NAME=Restaurant/Business Name
UPI_CURRENCY=INR  # or LKR, USD, etc.
```

---

## 🎮 Usage Guide

### Menu Navigation
1. **Hover over items**: Move your hand to browse the menu
2. **Select category**: Pinch your fingers to select a category
3. **Browse items**: Use hand movement to scroll through items
4. **Add to cart**: Pinch gesture on desired item to add it

### Cart Management
- **View cart**: Items appear in a side panel
- **Adjust quantity**: Use gesture controls to increase/decrease quantity
- **Remove items**: Pinch on cart item to remove

### Checkout Process
1. Review your cart total
2. Confirm order with pinch gesture
3. QR code will be displayed for payment
4. Complete UPI payment using your phone
5. Receipt will be automatically generated and saved

### Voice Commands
The system provides audio feedback for:
- Menu category selection
- Item additions/removals
- Payment requests
- Order confirmation

---

## 💾 Output Files

### Receipt Format
CSV file with itemized details:
- Item name, quantity, price
- Subtotal and GST calculation
- Final total amount
- Timestamp of transaction

### Order Log
Backend log file tracks:
- All orders placed
- Transaction amounts
- Timestamp and duration
- User interaction metrics

---

## 🎨 Theme Colors

The application uses a modern neon color scheme:

```python
{
    "bg": (10, 10, 18),              # Dark background
    "header": (18, 18, 30),          # Header color
    "accent": (255, 150, 60),        # Orange accent
    "neon_green": (0, 255, 160),     # Neon green
    "neon_blue": (80, 190, 255),     # Neon blue
    "text_dark": (20, 20, 30),       # Dark text
    "muted": (120, 120, 140),        # Muted color
}
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Ensure webcam is connected and not in use by another application |
| Low gesture detection | Improve lighting conditions, stay within 1-2 meters of camera |
| Audio not working | Install pyttsx3 with: `pip install --upgrade pyttsx3` |
| QR code not generating | Verify UPI configuration in `.env` file |
| Slow performance | Reduce resolution or disable animations |

---

## 📊 System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|------------|
| **Python Version** | 3.8 | 3.10+ |
| **RAM** | 2 GB | 4+ GB |
| **CPU** | Dual-core 2.0 GHz | Quad-core 2.5+ GHz |
| **Webcam Resolution** | 640×480 @ 30fps | 1280×720 @ 30fps |
| **Storage** | 500 MB | 1+ GB |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Wafry Ahamed**  
GitHub: [@WafryAhamed](https://github.com/WafryAhamed)

---

## 🙏 Acknowledgments

- **MediaPipe** for advanced hand tracking
- **OpenCV** for computer vision capabilities
- **pyttsx3** for text-to-speech functionality
- All contributors and supporters

---


<div align="center">


⭐ If you find this project useful, please consider giving it a star!

</div>
