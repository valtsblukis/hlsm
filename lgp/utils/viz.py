import cv2
import numpy as np


def resize_to_width(figure, width, nearest=False):
    aspect = width / float(figure.shape[1])
    height = int(figure.shape[0] * aspect)
    img = cv2.resize(figure,
                     dsize=(width, height),
                     interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    return img


def resize_to_height(figure, height, nearest=False):
    aspect = height / float(figure.shape[0])
    width = int(figure.shape[1] * aspect)
    img = cv2.resize(figure,
                     dsize=(width, height),
                     interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    return img


def resize(figure, height, width, nearest=False):
    img = cv2.resize(figure,
                     dsize=(width, height),
                     interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    return img


def hstack(frames, height=None):
    if height is None:
        height = max([img.shape[0] for img in frames])
    frames = [resize_to_height(frame, height, nearest=False) for frame in frames]
    joined_frame = np.concatenate(frames, axis=1)
    return joined_frame


def vstack(frames, width=None):
    if width is None:
        width = max([img.shape[1] for img in frames])
    frames = [resize_to_width(frame, width, nearest=False) for frame in frames]
    joined_frame = np.concatenate(frames, axis=0)
    return joined_frame


def b_unify_size(frames, height=None, width=None):
    if width is None:
        width = max([img.shape[1] for img in frames])
    if height is None:
        height = max([img.shape[0] for img in frames])
    frames = [resize(frame, height, width, nearest=False) for frame in frames]
    return frames


def canvas_to_img(fig):
    fig.canvas.draw()
    canvas = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    canvas = canvas.reshape([int(fig.canvas.renderer.height), int(fig.canvas.renderer.width), 3])
    return canvas


def prep_image(image, scale=(1.0, 1.0), no_norm=False):
    import cv2
    if type(scale) is int or type(scale) is float:
        scale = (scale, scale)

    is_torch = hasattr(image, "cpu")
    if is_torch:
        image = image.detach().cpu().numpy()
        image = image.squeeze()
        if len(image.shape) == 3:
            image = image.transpose((1, 2, 0))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not no_norm:
        image = image - np.min(image)
        image = image / (np.max(image) + 1e-9)
    else:
        image = np.clip(image, 0.0, 1.0)

    # Only 2 channels - add another one
    if len(image.shape) == 3 and image.shape[2] == 2:
        newshape = list(image.shape)
        newshape[2] = 3
        new_img = np.zeros(newshape)
        new_img[:, :, 0:2] = image
        image = new_img

    # If we have too many channels, only show 3 of them
    if len(image.shape) > 2 and image.shape[2] > 3:
        image = image[:, :, 0:3]

    if scale != 1.0:
        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(scale[0] * width), int(scale[1] * height)), interpolation=cv2.INTER_NEAREST)

    if image.dtype == np.float64:
        image = image.astype(np.float32)

    # if len(image.shape) > 2 and (image.shape[2] == 3 or image.shape[2] == 4):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def show_image(image, name="live", waitkey=False, scale=1.0):
    import cv2
    image = prep_image(image, scale)

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow(name, image)
    if type(waitkey) is int:
        cv2.waitKey(waitkey)
    elif waitkey:
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)