import cv2

# Load the background image
img = cv2.imread("data/background.png")

# Resize window for better view
window_name = 'Click Top-Left and Bottom-Right Corners of Pink Box'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

points = []

# Mouse callback to get the points
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: {x}, {y}")
        if len(points) == 2:
            cv2.rectangle(img, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow(window_name, img)

# Show image and set callback
cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, click_event)

print("\n👉 Click on the **top-left corner** of the pink box, then the **bottom-right corner**.")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Print box size
if len(points) == 2:
    x1, y1 = points[0]
    x2, y2 = points[1]
    width = x2 - x1
    height = y2 - y1
    print(f"\n✅ Final Box Info:\nTop-left: ({x1}, {y1})\nWidth: {width}\nHeight: {height}")
else:
    print("❌ You didn't click two points.")
