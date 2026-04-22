# 🎮 FlameX7 AirBurger - Features & Capabilities

## 🌟 Core Features

### 1. Gesture Recognition System
- **Hand Detection**: Real-time hand tracking using MediaPipe
- **Joint Tracking**: Precise finger and hand joint detection
- **Pinch Gesture**: Thumb-to-index finger pinch for selections
- **Hover Hover Detection**: Item selection through proximity
- **Smooth Tracking**: AI-powered hand movement smoothing (α=0.65)

### 2. Menu System

#### Signature Burgers (₹360-₹450)
- 🍔 FlameX7 Prime Ember - ₹420.00
- 🍔 Nebula Crunch Stack - ₹390.00
- 🍔 Quantum Double Blaze - ₹450.00
- 🍔 Aero Veggie Flux - ₹360.00
- 🍔 HyperCheese Core Melt - ₹410.00

#### Sides & Add-ons (₹40-₹180)
- 🍟 Cosmic Fries Bucket - ₹160.00
- 🧅 Planet Rings (Onion) - ₹140.00
- 🧀 Astro Cheese Shots - ₹180.00
- 🥄 Star Dust Mayo Dip - ₹40.00

#### Drinks (₹70-₹130)
- 🥤 Plasma Cola - ₹80.00
- 🥤 Ion Lemon Fizz - ₹70.00
- 🥤 Dark Matter Cold Brew - ₹130.00

### 3. Shopping Cart
- **Dynamic Cart**: Real-time item addition/removal
- **Quantity Management**: Adjust quantities with gestures
- **Live Totals**: Instant subtotal, GST, and total calculation
- **Cart Visualization**: Visual cart display with item details
- **Item Images**: Burger images display in cart preview

### 4. Payment System

#### UPI Integration
- **QR Code Generation**: Dynamic UPI-based QR codes
- **Payment Gateway**: Direct integration with UPI protocol
- **Multiple Currencies**: Support for LKR, INR, USD, etc.
- **Merchant Details**: Configurable via `.env` file

#### Tax Calculation
- **GST Support**: 5% GST automatic calculation
- **Flexible Rates**: Configurable tax percentages
- **Itemized Breakdown**: Clear subtotal, tax, and total display

### 5. Receipt Management

#### Digital Receipts
- **CSV Format**: Standard CSV receipt generation
- **Itemized Details**: 
  - Item name, quantity, unit price
  - Line subtotals
  - Subtotal calculation
  - GST amount
  - Final total
- **Timestamped**: Each receipt includes generation timestamp
- **Auto-save**: Receipts automatically saved with date-time naming

#### Order Logging
- **Backend Log**: `backend_order_log.csv` tracks all orders
- **Metrics**: Captures transaction amounts and timing
- **Analytics Ready**: Data structured for business analytics

### 6. User Experience

#### Voice Assistance
- **Text-to-Speech**: Real-time audio feedback
- **Configurable Rate**: Adjustable speech speed (default: 160 WPM)
- **Volume Control**: Adjustable audio volume (0.9 default)
- **Async Audio**: Non-blocking voice feedback
- **Multi-threading**: Safe concurrent audio operations

#### Visual Feedback
- **Status Messages**: Real-time on-screen feedback
- **Color Coding**: Neon-themed visual indicators
- **Animated UI**: Smooth animations (14 frames @ 60 FPS)
- **Item Highlighting**: Visual selection feedback

#### Idle Detection
- **Auto-timeout**: 60-second idle detection
- **Screen Reset**: Automatic return to home screen
- **User Protection**: Prevents accidental selections

### 7. Display Capabilities

#### Multi-Display Support
- **Main Display**: 1000×640 resolution
- **OLED Display**: 320×220 secondary display support
- **Resolution Scalable**: Configurable display parameters
- **Responsive Design**: Adapts to different screen sizes

#### Theme System
- **Modern Neon Theme**: Contemporary color scheme
- **Dark Background**: Eye-friendly dark mode (10,10,18)
- **Accent Colors**: 
  - Orange accent (255, 150, 60)
  - Neon green (0, 255, 160)
  - Neon blue (80, 190, 255)
- **Glassmorphism**: Glass-effect UI elements

### 8. Performance Optimization

#### Image Preloading
- **Cached Burger Images**: Pre-loaded at startup
- **Fast Rendering**: Reduced real-time processing
- **Efficient Memory**: Optimized image handling

#### Hand Tracking Optimization
- **Threshold Tuning**: 
  - Hover threshold: 1.3
  - Pinch distance: 35 pixels
- **Smooth Alpha**: 0.65 for stable tracking
- **Frame-rate Optimized**: 0.016s per frame (60 FPS)

### 9. Data Management

#### Configuration
- **Environment Variables**: `.env` file support
- **Dynamic Settings**: Load from `.env` at runtime
- **Secure Credentials**: UPI ID stored securely

#### Logging & Analytics
- **Transaction Logging**: All orders logged
- **Receipt History**: Individual receipt files
- **CSV Format**: Standard spreadsheet format
- **Data Persistence**: Permanent local storage

### 10. Technical Capabilities

#### Computer Vision
- **OpenCV 4.0+**: Professional image processing
- **Real-time Processing**: Live camera feed handling
- **Frame-by-frame Analysis**: Gesture detection per frame
- **Color Space Handling**: Multiple color format support

#### Machine Learning
- **MediaPipe ML Pipeline**: State-of-the-art hand detection
- **TensorFlow Backend**: Powerful ML inference
- **Multi-hand Support**: Detect multiple hands simultaneously
- **Landmark Detection**: 21-point hand skeleton tracking

---

## 🔧 Configurable Parameters

### Gesture Settings
```python
HOVER_THRESHOLD = 1.3         # Sensitivity to item selection
PINCH_DIST_PX = 35            # Pinch gesture trigger distance
```

### Display Settings
```python
WIDTH, HEIGHT = 1000, 640     # Main display resolution
OLED_W, OLED_H = 320, 220    # Secondary display size
```

### Animation Settings
```python
ANIM_FRAMES = 14              # Animation frame count
ANIM_FPS_DELAY = 0.016        # Delay between frames (60 FPS)
SMOOTH_ALPHA = 0.65           # Hand tracking smoothing
```

### Behavior Settings
```python
IDLE_TIMEOUT = 60.0           # Idle timeout in seconds
PINCH_DIST_PX = 35            # Pinch sensitivity
```

### Tax Settings
```python
GST_PERCENT = 5.0             # GST percentage (default: 5%)
```

---

## 📈 Scalability Features

- **Multi-language Support Ready**: Voice engine supports multiple languages
- **Currency Flexibility**: Configurable via `.env`
- **Extensible Menu**: Easy to add new items
- **Plugin-ready Architecture**: Modular design for extensions
- **Cloud Integration Ready**: APIs structured for backend services

---

## 🔒 Security Features

- **Environment Variables**: Sensitive data in `.env`
- **No Hardcoded Credentials**: Configuration-driven
- **Input Validation**: Safe UPI ID handling
- **Thread-safe Operations**: Concurrent audio processing

---

## 🚀 Future Enhancement Possibilities

- [ ] Multiple language support
- [ ] Customer loyalty program
- [ ] Analytics dashboard
- [ ] Mobile app integration
- [ ] Cloud synchronization
- [ ] Advanced gesture recognition (3D)
- [ ] AR menu display
- [ ] Real-time inventory management

---

**All features tested and verified! System ready for deployment. ✓**
