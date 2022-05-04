import cv2 as cv
import numpy as np
import math
import os
import random
import numpy as np
from salience import crop_img_to_mask

# region Initialisations
# Scaling
scalar = 8
imageScale = 4

# Fonts
headline_font_scale = 0.025 * scalar * imageScale
headline_line_height = int(0.3 * scalar * imageScale)
headline_font = cv.FONT_HERSHEY_TRIPLEX

text_font_scale = 0.0125 * scalar * imageScale
text_line_height = int(0.15 * scalar * imageScale)
text_font = cv.FONT_HERSHEY_SIMPLEX
# endregion


def clamp_bbox(bbox, img):
    """Clamp a bbox's bounds to the bounds of an image."""
    bbox[0] = int(max(0, bbox[0]))
    bbox[1] = int(max(0, min(img.shape[1], bbox[1])))
    bbox[2] = int(max(0, bbox[2]))
    bbox[3] = int(max(0, min(img.shape[0], bbox[3])))

    return bbox


def flatten_list(t):
    """Flatten a list of lists."""
    return [item for sublist in t for item in sublist]


def fill_bbox(img, bbox, val):
    """Fill a bbox of an image with a value."""
    img[bbox[2]:bbox[3],
        bbox[0]:bbox[1]] = val


def fill_bbox_with_img(img, bbox, val):
    """Fill a bounding box with an image."""
    try:
        h = val.shape[0]
        w = val.shape[1]

        img[bbox[2]:bbox[2]+h,
            bbox[0]:bbox[0]+w] = val
    except Exception as e:
        pass


def fill_bbox_with_text(img, bbox, font_scale, line_height, font):
    """Fill a bounding box with text."""

    # region Generate lipsum
    lipsumss = [["The words hadn't flowed from his fingers for the past few weeks. He never imagined he'd find himself with writer's block, but here he sat with a blank screen in front of him. That blank screen taunting him day after day had started to play with his mind. He didn't understand why he couldn't even type a single word, just one to begin the process and build from there. And yet, he already knew that the eight hours he was prepared to sit in front of his computer today would end with the screen remaining blank. \n",
                "He stared out the window at the snowy field. He'd been stuck in the house for close to a month and his only view of the outside world was through the window. There wasn't much to see. It was mostly just the field with an occasional bird or small animal who ventured into the field. As he continued to stare out the window, he wondered how much longer he'd be shackled to the steel bar inside the house. \n",
                 "A long black shadow slid across the pavement near their feet and the five Venusians, very much startled, looked overhead. They were barely in time to see the huge gray form of the carnivore before it vanished behind a sign atop a nearby building which bore the mystifying information Pepsi-Cola. \n",
                 ], ["It was so great to hear from you today and it was such weird timing, he said. This is going to sound funny and a little strange, but you were in a dream I had just a couple of days ago. I'd love to get together and tell you about it if you're up for a cup of coffee, he continued, laying the trap he'd been planning for years. \n",
                     "Dave watched as the forest burned up on the hill, only a few miles from her house. The car had been hastily packed and Marta was inside trying to round up the last of the pets. Dave went through his mental list of the most important papers and documents that they couldn't leave behind. He scolded himself for not having prepared these better in advance and hoped that he had remembered everything that was needed. He continued to wait for Marta to appear with the pets, but she still was nowhere to be seen. \n",
                     "Sometimes it's just better not to be seen. That's how Harry had always lived his life. He prided himself as being the fly on the wall and the fae that blended into the crowd. That's why he was so shocked that she noticed him. \n",
                     ], ["Betty was a creature of habit and she thought she liked it that way. That was until Dave showed up in her life. She now had a choice to make and it would determine whether her lie remained the same or if it would change forever. \n",
                         "He scolded himself for being so tentative. He knew he shouldn't be so cautious, but there was a sixth sense telling him that things weren't exactly as they appeared. It was that weird chill that rolls up your neck and makes the hair stand on end. He knew that being so tentative could end up costing him the job, but he learned that listening to his sixth sense usually kept him from getting into a lot of trouble. \n",
                         "One foot in front of the other, One more step, and then one more. Jack's only thoughts were to keep moving no matter how much his body screamed to stop and rest. He's lost almost all his energy and his entire body ached beyond belief, but he forced himself to take another step. Then another. And then one more. \n",
                         ], ["She had come to the conclusion that you could tell a lot about a person by their ears. The way they stuck out and the size of the earlobes could give you wonderful insights into the person. Of course, she couldn't scientifically prove any of this, but that didn't matter to her. Before anything else, she would size up the ears of the person she was talking to. \n",
                             "It wasn't quite yet time to panic. There was still time to salvage the situation. At least that is what she was telling himself. The reality was that it was time to panic and there wasn't time to salvage the situation, but he continued to delude himself into believing there was. \n",
                             "It's always good to bring a slower friend with you on a hike. If you happen to come across bears, the whole group doesn't have to worry. Only the slowest in the group do. That was the lesson they were about to learn that day. \n",
                             ], ["The seekers would come into the emergency room and scream about how much pain they were in. When you told them that you would start elevating their pain with Tylenol or Advil instead of a narcotic they became nasty and combative. They would start telling you what and dose they had to have to make their pain tolerable. After dealing with the same seekers several times a month it gets old. Some of the doctors would give in and give them a dose of morphine and send them away. Sure that was faster, but ethically she still couldn’t do it. Perhaps that’s why she had longer care times than the other doctors. \n",
                                 "He sat staring at the person in the train stopped at the station going in the opposite direction. She sat staring ahead, never noticing that she was being watched. Both trains began to move and he knew that in another timeline or in another universe, they had been happy together. \n",
                                 "All he wanted was a candy bar. It didn't seem like a difficult request to comprehend, but the clerk remained frozen and didn't seem to want to honor the request. It might have had something to do with the gun pointed at his face. \n",
                                 ], ["If you can imagine a furry humanoid seven feet tall, with the face of an intelligent gorilla and the braincase of a man, you'll have a rough idea of what they looked like -- except for their teeth. The canines would have fitted better in the face of a tiger, and showed at the corners of their wide, thin-lipped mouths, giving them an expression of ferocity. \n",
                                     "The boy walked down the street in a carefree way, playing without notice of what was about him. He didn't hear the sound of the car as his ball careened into the road. He took a step toward it, and in doing so sealed his fate. \n",
                                     "She counted. One. She could hear the steps coming closer. Two. Puffs of breath could be seen coming from his mouth. Three. He stopped beside her. Four. She pulled the trigger of the gun. \n",
                                     ], ["Lori lived her life through the lens of a camera. She never realized this until this very moment as she scrolled through thousands of images on your computer. She could remember the exact moment each photo was taken. She could remember where she had been, what she was thinking as she tried to get the shot, the smells of the surrounding area, and even the emotions that she felt taking the photo, yet she had trouble remembering what she had for breakfast. \n",
                                         "Yes, she was now the first person ever to land on Venus, but that was of little consequence. Her name would be read by millions in school as the first to land here, but that celebrity would never actually be seen by her. She looked at the control panel and knew there was nothing that would ever get it back into working order. She was the first and it was not clear this would also be her last. \n",
                                         ]]

    lipsums = random.choice(lipsumss)
    random.shuffle(lipsums)
    lipsum = flatten_list(lipsums)
    # endregion

    # Create empty text image
    if bbox[3]-bbox[2] <= 0 or bbox[1]-bbox[0] <= 0:
        return
    text_img = np.ones(
        (bbox[3]-bbox[2], bbox[1]-bbox[0], 3), np.uint8) * 255

    # Calculate the number of lines of text needed to fill the bounding box
    (w_W, h_W), _ = cv.getTextSize(
        "w", font, font_scale, 1)
    num_lines = math.floor(text_img.shape[0]/(h_W+line_height))

    char_pointer = 0

    # Fill the bounding box with text
    for i in range(num_lines):
        # Initialise empty line
        line = ""
        line_size = 0

        # Implement line wrapping. Add a character to the line while the length of the line is less than the bounding box's shape.
        while line_size < text_img.shape[1]:
            if(lipsum[char_pointer] == " " and line_size == 0):
                pass
            else:
                line += lipsum[char_pointer]
                (line_size, _), _ = cv.getTextSize(
                    line, font, font_scale, 1)
            char_pointer += 1
            char_pointer = char_pointer % len(lipsum)

        # Avoid text overflow
        char_pointer -= 1
        line = line[:-1]

        # Draw the line
        bottom = (h_W*i)+line_height*(i+2)
        cv.putText(text_img, line,
                   (0, bottom),
                   font,
                   font_scale,
                   (0, 0, 0),
                   1,
                   2)

    # Crop val to the size of the bounding box
    fill_bbox(img, bbox, text_img)


def render(bboxes):
    """Generate a rendering of bounding boxes using our image cropping algorithm."""

    # Convert to numpy
    bboxes["text"] = np.array(bboxes["text"])
    bboxes["headline"] = np.array(bboxes["headline"])
    bboxes["image"] = np.array(bboxes["image"])

    # Initialise output
    output = np.ones((300 * imageScale, 225 * imageScale, 3),
                     dtype=np.uint8) * 255

    # region Create mask
    mask = np.ones((32, 24))

    for img_bbox in bboxes["image"]:
        fill_bbox(mask, img_bbox, 0)

    for text_bbox in bboxes["text"]:
        fill_bbox(mask, text_bbox, 1)

    for headline_bbox in bboxes["headline"]:
        fill_bbox(mask, headline_bbox, 1)
    # endregion

    # Load images and initialise counter
    img_counter = 0
    imgs = []
    for f in os.listdir("./images/"):
        ext = os.path.splitext(f)[1]
        if ext.lower() in [".jpg", ".png"]:
            imgs.append("./images/"+f)
    random.shuffle(imgs)

    # Add images
    for img_bbox in bboxes["image"]:
        # Crop images to the mask appropriately using image saliency
        img_mask = mask[img_bbox[2]:img_bbox[3], img_bbox[0]:img_bbox[1]]

        img = crop_img_to_mask(
            imgs[img_counter], img_mask, math.floor(300/32) * imageScale, drawFocal=False)

        img_bbox = img_bbox*math.floor(300/32)*imageScale
        img_bbox = clamp_bbox(img_bbox, output)
        fill_bbox_with_img(output, img_bbox, img)

        # Increment counter
        img_counter += 1
        img_counter = img_counter % len(imgs)

    # Add text
    for text_bbox in bboxes["text"] * math.floor(300/32) * imageScale:
        text_bbox = clamp_bbox(text_bbox, output)

        fill_bbox_with_text(output, text_bbox, text_font_scale,
                            text_line_height, text_font)

    # Add headlines
    for headline_bbox in bboxes["headline"] * math.floor(300/32) * imageScale:
        headline_bbox = clamp_bbox(headline_bbox, output)

        fill_bbox_with_text(output, headline_bbox,
                            headline_font_scale, headline_line_height, headline_font)

    return output


def render_b2(bboxes):
    """Generate a rendering of bounding boxes without our image cropping algorithm, i.e. use basic central image cropping."""

    # Convert to numpy arrays
    bboxes["text"] = np.array(bboxes["text"])
    bboxes["headline"] = np.array(bboxes["headline"])
    bboxes["image"] = np.array(bboxes["image"])

    output = np.ones((300 * imageScale, 225 * imageScale, 3),
                     dtype=np.uint8) * 255

    # Load images and initialise counter
    img_counter = 0
    imgs = []
    for f in os.listdir("./images/"):
        ext = os.path.splitext(f)[1]
        if ext.lower() in [".jpg", ".png"]:
            imgs.append("./images/"+f)
    random.shuffle(imgs)

    # Add images
    for img_bbox in bboxes["image"]:
        # Central crop images to the bounding box
        img_bbox = clamp_bbox(img_bbox*math.floor(300/32)*imageScale, output)
        w = img_bbox[1] - img_bbox[0]
        h = img_bbox[3] - img_bbox[2]

        img = cv.imread(imgs[img_counter])

        (h_img, w_img) = img.shape[:2]

        # Ratio < 1 means portrait
        slot_ratio = w/h
        img_ratio = w_img/h_img

        # Resize image
        if(slot_ratio > img_ratio):
            # Ensure that widths match
            r = w/float(w_img)
            img = cv.resize(
                img, (w, math.ceil(h_img*r)))
        else:
            # Ensure that heights match
            r = h/float(h_img)
            img = cv.resize(
                img, (math.ceil(w_img*r), h))

        # Crop image centrally
        # If landscape slot
        if (slot_ratio > 1):
            # Crop centrally vertical
            (h_img, w_img) = img.shape[:2]
            center = h_img/2
            top = math.floor(center-h/2)
            bottom = math.floor(center+h/2)
            img = img[top:bottom, 0:w]
        # If portrait slot
        else:
            # Crop centrally horizontal
            (h_img, w_img) = img.shape[:2]
            center = w_img/2
            left = math.floor(center-w/2)
            right = math.floor(center+w/2)
            img = img[0:h, left:right]

        fill_bbox_with_img(output, img_bbox, img)

        # Increment counter
        img_counter += 1
        img_counter = img_counter % len(imgs)

    # Add text
    for text_bbox in bboxes["text"] * math.floor(300/32) * imageScale:
        text_bbox = clamp_bbox(text_bbox, output)

        fill_bbox_with_text(output, text_bbox, text_font_scale,
                            text_line_height, text_font)

    # Add headlines
    for headline_bbox in bboxes["headline"] * math.floor(300/32) * imageScale:
        headline_bbox = clamp_bbox(headline_bbox, output)

        fill_bbox_with_text(output, headline_bbox,
                            headline_font_scale, headline_line_height, headline_font)

    return output
