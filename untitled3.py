# import cv2
# import numpy as np

# # Global variables to store the image and drawing state
# drawing = False
# edited_image = None
# original_image = None

# def draw_filled_region(event, x, y, flags, param):
#     """
#     Mouse callback function to draw on the image.
#     """
#     global drawing, edited_image

#     # Check for the left mouse button press event
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         # Draw a filled circle at the cursor position
#         cv2.circle(edited_image, (x, y), 5, (0, 0, 0), -1)

#     # Check for mouse movement while the left button is held down
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             # Draw a filled circle to simulate a brush stroke
#             cv2.circle(edited_image, (x, y), 5, (0, 0, 0), -1)

#     # Check for the left mouse button release event
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         # Finalize the stroke with a filled circle
#         cv2.circle(edited_image, (x, y), 5, (0, 0, 0), -1)

# def main():
#     """
#     Main function to load an image, set up the interactive window, and handle user input.
#     """
#     global edited_image, original_image

#     # Replace with the path to your occupancy map image
#     image_path = '/data/datasets/umgloc_dataset/b5l5_4_rev.pgm'

#     # Load the image in grayscale
#     original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     if original_image is None:
#         print("Error: Could not load image from", image_path)
#         return

#     # Create a copy of the original image to work on
#     edited_image = original_image.copy()

#     # Create a window to display the image
#     cv2.namedWindow('Interactive Map Editor')

#     # Set the mouse callback function for the window
#     cv2.setMouseCallback('Interactive Map Editor', draw_filled_region)

#     print("Interactive Map Editor:")
#     print(" - Press and hold the left mouse button to paint and remove noise.")
#     print(" - Press 's' to save the edited map.")
#     print(" - Press 'q' to quit without saving.")

#     while True:
#         # Display the current state of the edited image
#         cv2.imshow('Interactive Map Editor', edited_image)

#         # Wait for a key press
#         key = cv2.waitKey(1) & 0xFF

#         # If the 's' key is pressed, save the image and quit
#         if key == ord('s'):
#             save_path = 'edited_occupancy_map.png'
#             cv2.imwrite(save_path, edited_image)
#             print(f"Map saved to {save_path}")
#             break

#         # If the 'q' key is pressed, quit
#         elif key == ord('q'):
#             print("Exiting without saving.")
#             break

#     # Close all OpenCV windows
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw

# # Global variables to store the image and drawing state
# drawing = False
# image_path = 'data/datasets/umgloc_dataset/b5l5_4_rev.pgm'
# edited_image = None
# ax = None
# fig = None

# def on_press(event):
#     """
#     Mouse press event handler.
#     """
#     global drawing
#     if event.button == 1: # Left mouse button
#         drawing = True

# def on_release(event):
#     """
#     Mouse release event handler.
#     """
#     global drawing
#     if event.button == 1: # Left mouse button
#         drawing = False

# def on_motion(event):
#     """
#     Mouse motion event handler.
#     """
#     global drawing, edited_image, ax
#     if drawing and event.xdata is not None and event.ydata is not None:
#         # Get coordinates and create a Pillow ImageDraw object
#         x, y = int(round(event.xdata)), int(round(event.ydata))
#         pil_image = Image.fromarray(edited_image)
#         draw = ImageDraw.Draw(pil_image)
#         # Use the ImageDraw object to "paint" a black circle (radius=3)
#         draw.ellipse((x-3, y-3, x+3, y+3), fill=0)
#         # Convert the Pillow image back to a NumPy array for matplotlib
#         edited_image = np.array(pil_image)
#         # Update the displayed image
#         ax.imshow(edited_image, cmap='gray')
#         fig.canvas.draw_idle()

# def save_image(image, file_path):
#     """
#     Saves the image using Pillow.
#     """
#     pil_image = Image.fromarray(image)
#     pil_image.save(file_path)
#     print(f"Image saved to {file_path}")

# def main():
#     """
#     Main function to set up the interactive editor.
#     """
#     global edited_image, ax, fig

#     try:
#         original_image = Image.open(image_path).convert('L') # Load in grayscale
#         edited_image = np.array(original_image)
#     except FileNotFoundError:
#         print(f"Error: Image not found at {image_path}")
#         return

#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(edited_image, cmap='gray')
#     ax.set_title('Interactive Map Editor (Use mouse to draw, close window to quit)')

#     # Connect mouse events to the callback functions
#     fig.canvas.mpl_connect('button_press_event', on_press)
#     fig.canvas.mpl_connect('button_release_event', on_release)
#     fig.canvas.mpl_connect('motion_notify_event', on_motion)

#     plt.show()

#     # After the window is closed, save the edited image
#     if edited_image is not None:
#         save_image(edited_image, 'edited_occupancy_map_pil.png')

# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.patches as patches

# Global variables to store the image, drawing state, and rectangle coordinates
image_path = 'data/datasets/umgloc_dataset/b5l5_4_rev.pgm'
edited_image = None
ax = None
fig = None
rect_patch = None
start_x, start_y = None, None
is_drawing = False  # only True when Shift+LMB is used

def _shift_held(event) -> bool:
    """Return True iff Shift is pressed for this event."""
    # Matplotlib sets event.key to 'shift' (or contains 'shift' for combos).
    return isinstance(event.key, str) and ('shift' in event.key.lower())

def on_press(event):
    """Mouse press: start drawing only if Shift + left button."""
    global start_x, start_y, rect_patch, is_drawing
    if event.button == 1 and event.inaxes and _shift_held(event):
        is_drawing = True
        start_x, start_y = event.xdata, event.ydata
        # Create a temporary rectangle for feedback
        rect_patch = patches.Rectangle((start_x, start_y), 0, 0,
                                       linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect_patch)
        fig.canvas.draw_idle()

def on_motion(event):
    """Mouse motion: update rectangle only while drawing (Shift was held on press)."""
    global start_x, start_y, rect_patch, is_drawing
    if not is_drawing or rect_patch is None or not event.inaxes:
        return
    x, y = event.xdata, event.ydata
    # Keep rectangle well-defined with positive width/height
    x0, y0 = min(start_x, x), min(start_y, y)
    w, h = abs(x - start_x), abs(y - start_y)
    rect_patch.set_xy((x0, y0))
    rect_patch.set_width(w)
    rect_patch.set_height(h)
    fig.canvas.draw_idle()

def on_release(event):
    """Mouse release: commit the filled rectangle if we were drawing."""
    global start_x, start_y, edited_image, rect_patch, is_drawing
    if not is_drawing or event.button != 1 or not event.inaxes:
        return

    end_x, end_y = event.xdata, event.ydata

    # Remove temporary rectangle
    if rect_patch is not None:
        rect_patch.remove()
        rect_patch = None

    # Compute integer bounds
    x1, y1 = int(round(min(start_x, end_x))), int(round(min(start_y, end_y)))
    x2, y2 = int(round(max(start_x, end_x))), int(round(max(start_y, end_y)))

    if x2 > x1 and y2 > y1:
        pil_image = Image.fromarray(edited_image)
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle([(x1, y1), (x2, y2)], fill=0)   # fill with black
        edited_image = np.array(pil_image)
        ax.imshow(edited_image, cmap='gray')
        fig.canvas.draw_idle()

    # Reset state
    is_drawing = False

def main():
    global edited_image, ax, fig

    try:
        original_image = Image.open(image_path).convert('L')  # grayscale
        edited_image = np.array(original_image)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(edited_image, cmap='gray')
    ax.set_title('Interactive Editor — hold Shift + Left Mouse to draw')

    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show()

    if edited_image is not None:
        save_image(edited_image, 'edited_occupancy_map_rect.png')

def save_image(image, file_path):
    Image.fromarray(image).save(file_path)
    print(f"Image saved to {file_path}")

if __name__ == "__main__":
    main()

