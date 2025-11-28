import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
from datetime import datetime
import qrcode
from dotenv import load_dotenv
import os


load_dotenv()

# ---------------- CONFIG / TUNING ----------------
HOVER_THRESHOLD = 1.3   # larger => less sensitive (seconds to dwell)
PINCH_DIST_PX = 35      # distance between thumb & index for "click"
OLED_W, OLED_H = 320, 220

# Animation settings
ANIM_FRAMES = 14        # number of frames for slide animation
ANIM_FPS_DELAY = 0.016  # ~60 FPS

# Smoothing for fingertip position
SMOOTH_ALPHA = 0.65     # higher -> smoother (0..1)

# Base frame size
WIDTH, HEIGHT = 1000, 640

# UPI Payment Configuration (from .env with safe defaults)
UPI_ID = os.getenv("UPI_ID", "flamex7@upi")               # demo fallback
UPI_NAME = os.getenv("UPI_NAME", "FlameX7 AirBurger")
UPI_CURRENCY = os.getenv("UPI_CURRENCY", "LKR")

# ---------------- MENU (FlameX7 Signature Burgers) ----------------
menu = {
    "Signature Burgers": [
        ("FlameX7 Prime Ember", 420.00, "assets/burgers/burger1.png"),
        ("Nebula Crunch Stack", 390.00, "assets/burgers/burger2.png"),
        ("Quantum Double Blaze", 450.00, "assets/burgers/burger3.png"),
        ("Aero Veggie Flux", 360.00, "assets/burgers/burger4.png"),
        ("HyperCheese Core Melt", 410.00, "assets/burgers/burger5.png"),
    ],
    "Sides & Add-ons": [
        ("Cosmic Fries Bucket", 160.00, None),
        ("Planet Rings (Onion)", 140.00, None),
        ("Astro Cheese Shots", 180.00, None),
        ("Star Dust Mayo Dip", 40.00, None),
    ],
    "Drinks": [
        ("Plasma Cola", 80.00, None),
        ("Ion Lemon Fizz", 70.00, None),
        ("Dark Matter Cold Brew", 130.00, None),
    ]
}

# Preload burger images (only for items with paths)
BURGER_IMAGES = {}
for category, items in menu.items():
    for name, price, img_path in items:
        if img_path:
            img = cv2.imread(img_path)
            if img is not None:
                BURGER_IMAGES[name] = img

# ---------------- THEME (Futuristic Glass UI) ----------------
C = {
    "bg": (10, 10, 18),
    "header": (18, 18, 30),
    "glass_top": (245, 245, 255),
    "glass_bottom": (190, 200, 230),
    "accent": (255, 150, 60),       # flame orange
    "text_dark": (20, 20, 30),
    "muted": (120, 120, 140),
    "neon_green": (0, 255, 160),
    "neon_blue": (80, 190, 255),
}

# ---------------- STATE ----------------
state = {
    "screen": "home",               # "home", "categories", "items", "cart", "payment", "paid"
    "current_category": None,
    "cart": [],                     # list of {name, price, qty}
    "msg": "Welcome to FlameX7 AirBurger!",
    "last_total": 0.0,              # last paid total
}

# ---------------- HELPERS ----------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_pinch(lm):
    return lm and dist(lm[4], lm[8]) < PINCH_DIST_PX

def find_cart(name):
    for i, it in enumerate(state["cart"]):
        if it["name"] == name:
            return i
    return None

def compute_totals(gst_percent=5.0):
    subtotal = sum(i["qty"] * i["price"] for i in state["cart"])
    gst = subtotal * gst_percent / 100.0
    total = subtotal + gst
    return subtotal, gst, total

def save_receipt(cart, gst_percent=5.0):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"receipt_{now}.csv"
    subtotal = sum(i["qty"] * i["price"] for i in cart)
    gst = subtotal * gst_percent / 100.0
    total = subtotal + gst

    with open(fname, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["FlameX7 AirBurger – Touchless Order Receipt"])
        w.writerow(["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        w.writerow([])
        w.writerow(["Item", "Qty", "Price", "Subtotal"])
        for it in cart:
            sub = it["qty"] * it["price"]
            w.writerow([it["name"], it["qty"], f"{it['price']:.2f}", f"{sub:.2f}"])
        w.writerow([])
        w.writerow(["Subtotal", "", "", f"{subtotal:.2f}"])
        w.writerow(["GST (5%)", "", "", f"{gst:.2f}"])
        w.writerow(["TOTAL", "", "", f"{total:.2f}"])

    return fname, total

def simulate_backend_log(cart, total):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("backend_order_log.csv", "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ORDER_AT", now, "TOTAL", f"{total:.2f}", "ITEMS", len(cart)])
        for it in cart:
            w.writerow(["ITEM", it["name"], "QTY", it["qty"], "PRICE", f"{it['price']:.2f}"])
        w.writerow([])

def generate_upi_uri(amount):
    return (
        f"upi://pay?"
        f"pa={UPI_ID}"
        f"&pn={UPI_NAME.replace(' ', '%20')}"
        f"&am={amount:.2f}"
        f"&cu={UPI_CURRENCY}"
    )

def show_upi_qr(amount):
    uri = generate_upi_uri(amount)
    qr_img_pil = qrcode.make(uri)
    qr_img = np.array(qr_img_pil.convert("RGB"))[:, :, ::-1]  # PIL RGB -> OpenCV BGR
    qr_img = cv2.resize(qr_img, (260, 260))

    canvas = np.zeros((360, 360, 3), np.uint8)
    canvas[:] = C["bg"]
    x_offset = (canvas.shape[1] - qr_img.shape[1]) // 2
    y_offset = 40
    canvas[y_offset:y_offset+qr_img.shape[0], x_offset:x_offset+qr_img.shape[1]] = qr_img

    cv2.putText(canvas, "Scan to Pay via UPI", (50, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C["neon_green"], 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Total: {amount:.2f} {UPI_CURRENCY}", (60, 335),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("FlameX7 UPI QR", canvas)

def oled(text):
    img = np.zeros((OLED_H, OLED_W, 3), np.uint8)
    cv2.rectangle(img, (0, 0), (OLED_W - 1, OLED_H - 1), (120, 120, 140), 2)
    title = "FlameX7 OLED"
    cv2.putText(img, title, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 160), 2)

    words = text.split()
    lines = []
    line = ""
    for w in words:
        if len(line + " " + w) > 22:
            lines.append(line)
            line = w
        else:
            line += " " + w
    if line:
        lines.append(line)

    y = 60
    for ln in lines[:5]:  # max 5 lines
        cv2.putText(img, ln.strip(), (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 26

    cv2.imshow("FlameX7 OLED Status", img)

# ---------------- PREMIUM GLASS BUTTON (FROST) ----------------
def rounded_rect_mask(w, h, r):
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1)
    cv2.circle(mask, (w - r, h - r), r, 255, -1)
    return mask

def draw_glass_button(img, rect, text, hover=False):
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    radius = min(25, h // 2 - 2)

    if w <= 0 or h <= 0:
        return

    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return

    overlay = np.zeros_like(roi)
    top_col = np.array(C["glass_top"], dtype=np.uint8)
    bot_col = np.array(C["glass_bottom"], dtype=np.uint8)
    for i in range(h):
        alpha = i / max(1, h - 1)
        overlay[i, :, :] = (1 - alpha) * top_col + alpha * bot_col

    blur = cv2.GaussianBlur(roi, (25, 25), 0)
    glass = cv2.addWeighted(blur, 0.55, overlay, 0.45, 0)

    mask = rounded_rect_mask(w, h, radius)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) // 255

    region = img[y1:y2, x1:x2]
    np.copyto(region, region * (1 - mask_color) + glass * mask_color,
              where=(mask[:, :, None] > 0))

    border_col = (80, 80, 100)
    if hover:
        border_col = C["accent"]
        brighten = np.clip(region + 18, 0, 255).astype(np.uint8)
        np.copyto(region, np.where(mask[:, :, None] > 0, brighten, region))

    mask_uint = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img[y1:y2, x1:x2], contours, -1, border_col, 2, lineType=cv2.LINE_AA)

    if hover:
        glow = np.zeros_like(region, dtype=np.uint8)
        cv2.rectangle(glow, (8, 8), (w - 8, h - 8), (255, 240, 200), -1)
        glow = cv2.GaussianBlur(glow, (31, 31), 0)
        alpha = 0.08
        region[:] = cv2.addWeighted(region, 1.0, glow.astype(np.uint8), alpha, 0)

    # Adjust font scaling for better fit
    max_chars = 28
    if len(text) > max_chars:
        font_scale = 0.65
        max_chars = 32
    else:
        font_scale = 0.85

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    tx = x1 + max(15, (w - tw) // 2)
    ty = y1 + max(28, (h + th) // 2 - 4)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, C["text_dark"], 2, cv2.LINE_AA)

def draw_hover_circle(frame, pos, t):
    cx, cy = pos
    r_outer = 20 + int(5 * (0.5 + 0.5 * math.sin(t * 6)))
    cv2.circle(frame, (cx, cy), r_outer, C["neon_green"], 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 8, C["neon_green"], -1, lineType=cv2.LINE_AA)

# ---------------- RENDER PER SCREEN ----------------
def render_screen_base(frame, buttons, draw_extra=None):
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (WIDTH, 80), C["header"], -1)

    cv2.putText(canvas, "FlameX7 AirBurger", (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, C["neon_blue"], 4, cv2.LINE_AA)
    cv2.putText(canvas, "FlameX7 AirBurger", (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (10, 10, 20), 2, cv2.LINE_AA)

    for b in buttons:
        draw_glass_button(canvas, b["rect"], b["label"], hover=False)

    if draw_extra:
        draw_extra(canvas)

    total_items = sum(it["qty"] for it in state["cart"])
    total = sum(it["qty"] * it["price"] for it in state["cart"])

    cv2.putText(canvas,
                f"Items: {total_items}   Total: {total:.2f} {UPI_CURRENCY}",
                (30, HEIGHT - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas

def build_buttons_for_screen(screen):
    buttons = []
    if screen == "home":
        items = [
            ("menu", "Browse Menu"),
            ("cart", "View Cart"),
            ("payment", "Pay & Generate QR"),
            ("exit", "Exit"),
        ]
        bw, bh = 320, 110
        gap_x, gap_y = 60, 30
        start_y = 160
        cols = 2
        rows = 2
        total_width = cols * bw + (cols - 1) * gap_x
        start_x = (WIDTH - total_width) // 2

        idx = 0
        for r in range(rows):
            for c in range(cols):
                x1 = start_x + c * (bw + gap_x)
                y1 = start_y + r * (bh + gap_y)
                buttons.append({
                    "id": items[idx][0],
                    "label": items[idx][1],
                    "rect": (x1, y1, x1 + bw, y1 + bh)
                })
                idx += 1

    elif screen == "categories":
        cats = list(menu.keys()) + ["Back to Home"]
        sx = 100
        bw = WIDTH - 2 * sx
        bh = 65
        gap = 18
        sy = 140
        for i, c in enumerate(cats):
            y1 = sy + i * (bh + gap)
            buttons.append({
                "id": "home" if c == "Back to Home" else "cat",
                "label": c,
                "rect": (sx, y1, sx + bw, y1 + bh)
            })

    elif screen == "items":
        cat = state["current_category"] or "Signature Burgers"
        items = menu[cat]
        sx = 60
        bw = 560
        bh = 68
        gap = 14
        sy = 140
        for i, (name, price, img_path) in enumerate(items):
            label = f"{name}   {price:.2f}"
            buttons.append({
                "id": "add",
                "label": label,
                "meta": {"name": name, "price": price, "img_path": img_path},
                "rect": (sx, sy + i * (bh + gap), sx + bw, sy + i * (bh + gap) + bh)
            })
        # Action buttons on right
        right_x = WIDTH - 220
        buttons.append({"id": "categories", "label": "Back to Categories",
                        "rect": (right_x, 150, right_x + 200, 210)})
        buttons.append({"id": "cart", "label": "View Cart",
                        "rect": (right_x, 240, right_x + 200, 300)})

    elif screen == "cart":
        if not state["cart"]:
            # Show empty message
            pass
        else:
            sx = 60
            bw = 560
            bh = 68
            gap = 14
            sy = 140
            for i, it in enumerate(state["cart"]):
                label = f"{it['name']}   {it['price']:.2f}   Qty: {it['qty']}"
                rect = (sx, sy + i * (bh + gap), sx + bw, sy + i * (bh + gap) + bh)
                buttons.append({
                    "id": "row",
                    "label": label,
                    "rect": rect,
                    "meta": {"idx": i}
                })
                # +/- buttons
                plus_rect = (WIDTH - 140, sy + i * (bh + gap), WIDTH - 80, sy + i * (bh + gap) + 33)
                minus_rect = (WIDTH - 140, sy + i * (bh + gap) + 35, WIDTH - 80, sy + i * (bh + gap) + 68)
                buttons.append({"id": "plus", "label": "+", "rect": plus_rect, "meta": {"idx": i}})
                buttons.append({"id": "minus", "label": "-", "rect": minus_rect, "meta": {"idx": i}})

        # Bottom action buttons
        btn_w, btn_h = 240, 60
        btn_left = (WIDTH - 2 * btn_w - 30) // 2
        buttons.append({"id": "payment", "label": "Proceed to Payment",
                        "rect": (btn_left, HEIGHT - 110, btn_left + btn_w, HEIGHT - 50)})
        buttons.append({"id": "home", "label": "Back to Home",
                        "rect": (btn_left + btn_w + 30, HEIGHT - 110, btn_left + 2 * btn_w + 30, HEIGHT - 50)})

    elif screen == "payment":
        subtotal, gst, total = compute_totals()
        btn_w, btn_h = 400, 65
        btn_x = (WIDTH - btn_w) // 2
        buttons.append({"id": "confirm",
                        "label": f"Confirm & Show UPI QR ({total:.2f} {UPI_CURRENCY})",
                        "rect": (btn_x, 300, btn_x + btn_w, 300 + btn_h)})
        buttons.append({"id": "home",
                        "label": "Back to Home",
                        "rect": (btn_x, 390, btn_x + btn_w, 390 + btn_h)})

    elif screen == "paid":
        btn_w, btn_h = 280, 60
        btn_x = (WIDTH - btn_w) // 2
        buttons.append({"id": "home",
                        "label": "Back to Home",
                        "rect": (btn_x, 420, btn_x + btn_w, 420 + btn_h)})

    return buttons

def draw_extra_payment(canvas):
    subtotal, gst, total = compute_totals()
    y_offset = 140
    cv2.putText(canvas, f"Subtotal: {subtotal:.2f} {UPI_CURRENCY}",
                (WIDTH // 2 - 160, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"GST (5%): {gst:.2f} {UPI_CURRENCY}",
                (WIDTH // 2 - 160, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"TOTAL: {total:.2f} {UPI_CURRENCY}",
                (WIDTH // 2 - 160, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                C["neon_green"], 3, cv2.LINE_AA)

def draw_extra_items(canvas, buttons):
    for b in buttons:
        if b["id"] != "add":
            continue
        meta = b.get("meta", {})
        name = meta.get("name")
        img = BURGER_IMAGES.get(name)
        if img is None:
            continue

        rect = b["rect"]
        x1, y1, x2, y2 = rect
        thumb_h = y2 - y1 - 8
        thumb_w = int(thumb_h * 0.9)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        tx = x2 + 12
        ty = y1 + 4
        if tx + thumb_w <= WIDTH - 15:
            canvas[ty:ty+thumb_h, tx:tx+thumb_w] = thumb
            cv2.rectangle(canvas, (tx, ty),
                          (tx + thumb_w, ty + thumb_h),
                          (40, 40, 60), 1)

# ---------------- SLIDE-LEFT ANIMATION ----------------
def animate_slide_left(current_frame, next_frame, steps=ANIM_FRAMES):
    h, w = current_frame.shape[:2]
    for i in range(steps):
        t = (i + 1) / steps
        te = 1 - pow(1 - t, 3)
        dx = int(te * w)
        canvas = np.zeros_like(current_frame)
        x_new_start = w - dx
        if x_new_start < w:
            canvas[:, max(0, x_new_start):w] = next_frame[:, 0:min(w, dx)]
        x_old_end = w - dx
        if x_old_end > 0:
            canvas[:, 0:x_old_end] = current_frame[:, dx:w]
        cv2.imshow("FlameX7 AirBurger", canvas)
        if cv2.waitKey(int(ANIM_FPS_DELAY * 1000)) & 0xFF == 27:
            break

# ---------------- MAIN APP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

hover_label = None
hover_start = 0
prev_ix, prev_iy = 0, 0
last_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    lm = None
    if res.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)
        h, w = frame.shape[:2]
        lm = [(int(p.x * w), int(p.y * h))
              for p in res.multi_hand_landmarks[0].landmark]

    cur_screen = state["screen"]
    buttons = build_buttons_for_screen(cur_screen)

    if cur_screen == "payment":
        extra = lambda c: draw_extra_payment(c)
    elif cur_screen == "items":
        extra = lambda c: draw_extra_items(c, buttons)
    else:
        extra = None

    base_frame = render_screen_base(frame, buttons, extra)
    out_frame = base_frame.copy()

    tnow = time.time()
    dt = tnow - last_time if last_time else 0.03
    last_time = tnow

    if lm:
        raw_ix, raw_iy = lm[8]
        prev_ix = int(prev_ix * SMOOTH_ALPHA + raw_ix * (1 - SMOOTH_ALPHA))
        prev_iy = int(prev_iy * SMOOTH_ALPHA + raw_iy * (1 - SMOOTH_ALPHA))
        ix, iy = prev_ix, prev_iy

        cv2.circle(out_frame, (ix, iy), 5, C["neon_green"], -1)
        hovered = None

        for b in buttons:
            x1, y1, x2, y2 = b["rect"]
            if x1 < ix < x2 and y1 < iy < y2:
                hovered = b
                draw_glass_button(out_frame, b["rect"], b["label"], hover=True)
                draw_hover_circle(out_frame, (ix, iy), tnow)

                if hover_label != b["label"]:
                    hover_label = b["label"]
                    hover_start = time.time()
                else:
                    if (time.time() - hover_start > HOVER_THRESHOLD) or is_pinch(lm):
                        bid = b["id"]
                        prev_state = state["screen"]

                        if bid == "menu":
                            next_screen = "categories"
                            state["msg"] = "Browse FlameX7 categories with gestures."
                        elif bid == "cart":
                            next_screen = "cart"
                            state["msg"] = "Review and adjust your AirBurger cart."
                        elif bid == "payment":
                            next_screen = "payment"
                            state["msg"] = "Confirm your order and generate UPI QR."
                        elif bid == "exit":
                            cap.release()
                            cv2.destroyAllWindows()
                            raise SystemExit()
                        elif bid == "home":
                            next_screen = "home"
                            state["msg"] = "Welcome back to FlameX7 AirBurger."
                        elif bid == "cat":
                            state["current_category"] = b["label"]
                            next_screen = "items"
                            state["msg"] = f"Browsing: {b['label']}."
                        elif bid == "categories":
                            next_screen = "categories"
                            state["msg"] = "Choose a category to continue."
                        elif bid == "add":
                            m = b["meta"]
                            idx = find_cart(m["name"])
                            if idx is None:
                                state["cart"].append({
                                    "name": m["name"],
                                    "price": m["price"],
                                    "qty": 1
                                })
                            else:
                                state["cart"][idx]["qty"] += 1
                            next_screen = state["screen"]
                            state["msg"] = f"Added {m['name']} to cart."
                        elif bid == "plus":
                            idx = b["meta"]["idx"]
                            state["cart"][idx]["qty"] += 1
                            next_screen = state["screen"]
                            state["msg"] = f"Increased quantity of {state['cart'][idx]['name']}."
                        elif bid == "minus":
                            idx = b["meta"]["idx"]
                            state["cart"][idx]["qty"] -= 1
                            if state["cart"][idx]["qty"] <= 0:
                                removed = state["cart"].pop(idx)
                                state["msg"] = f"Removed {removed['name']} from cart."
                            else:
                                state["msg"] = f"Decreased quantity of {state['cart'][idx]['name']}."
                            next_screen = state["screen"]
                        elif bid == "confirm":
                            if not state["cart"]:
                                state["msg"] = "Cart is empty. Add items before payment."
                                next_screen = state["screen"]
                            else:
                                receipt_file, total = save_receipt(state["cart"])
                                simulate_backend_log(state["cart"], total)
                                state["last_total"] = total
                                show_upi_qr(total)
                                state["cart"].clear()
                                state["msg"] = "Order placed. Scan QR to pay. Thank you!"
                                next_screen = "paid"
                        elif bid == "row":
                            next_screen = state["screen"]
                        else:
                            next_screen = state["screen"]

                        if next_screen != prev_state:
                            ok2, frame2 = cap.read()
                            if not ok2:
                                frame2 = frame.copy()
                            frame2 = cv2.flip(frame2, 1)
                            frame2 = cv2.resize(frame2, (WIDTH, HEIGHT))
                            next_buttons = build_buttons_for_screen(next_screen)
                            if next_screen == "payment":
                                draw_extra2 = lambda c: draw_extra_payment(c)
                            elif next_screen == "items":
                                draw_extra2 = lambda c: draw_extra_items(c, next_buttons)
                            else:
                                draw_extra2 = None
                            next_frame = render_screen_base(frame2, next_buttons, draw_extra2)
                            animate_slide_left(out_frame, next_frame, steps=ANIM_FRAMES)
                            state["screen"] = next_screen
                        else:
                            tmp = out_frame.copy()
                            draw_glass_button(tmp, b["rect"], b["label"], hover=True)
                            cv2.imshow("FlameX7 AirBurger", tmp)
                            cv2.waitKey(80)

                        hover_start = 0
                        hover_label = None
                break

        if not hovered:
            hover_label = None
            hover_start = 0

    else:
        prev_ix = int(prev_ix * SMOOTH_ALPHA)
        prev_iy = int(prev_iy * SMOOTH_ALPHA)
        hover_label = None
        hover_start = 0

    oled(state["msg"])
    cv2.imshow("FlameX7 AirBurger", out_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()