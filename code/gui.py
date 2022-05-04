from PIL import ImageTk, Image
import tkinter as tk
import tkinter.font as tkFont
import cv2 as cv
import numpy as np
import generator
import refinement
import render
import salience
import math
from tkinter.filedialog import askopenfilename

# region Initialisations
window = tk.Tk()
window.title("Layout Generation")

# Scale of the gui windows
scale = 0.75
width = int(600*scale)
height = int(800*scale)

# Fonts
h1 = tkFont.Font(
    family="Lucida", size=14, weight="bold")
p = tkFont.Font(
    family="Lucida", size=10)

headline_font = tkFont.Font(
    family="Times New Roman", size=int(18*scale), weight="bold")
text_font = tkFont.Font(family="Times New Roman",
                        size=int(8*scale), weight="normal")

# Lipsum text
lipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas quis elit vel risus elementum imperdiet nec vitae nunc. Ut fringilla lobortis massa, quis vehicula purus semper sed. Nullam eu auctor felis, a congue ipsum. Fusce facilisis urna felis, eu luctus arcu fermentum condimentum. Maecenas euismod facilisis orci placerat lobortis. In convallis sapien sapien, facilisis pulvinar lorem placerat eu. Donec purus enim, luctus vel nibh et, iaculis consectetur nisl. Nunc vitae iaculis lectus, congue bibendum ante. Quisque efficitur vel urna ut tempus. Ut ac eros nibh. Quisque lectus purus, mollis vel condimentum sit amet, ultricies sit amet nisl. Etiam et purus vestibulum, vulputate nulla sit amet, cursus enim. Nunc sit amet quam congue, feugiat elit id, semper erat. Nullam vitae tempor sapien, a malesuada nisi. Cras posuere cursus felis, vel accumsan tortor. Sed nec tincidunt neque. Phasellus lectus justo, gravida sed tincidunt quis, pellentesque sit amet lacus. Maecenas nec ligula nibh. Phasellus sit amet purus euismod, auctor enim vel, vulputate arcu. In accumsan vehicula condimentum. Aenean dictum magna in ipsum ullamcorper malesuada. Nulla non posuere arcu, quis lobortis est. Suspendisse suscipit massa leo, eu mattis diam mattis et. Pellentesque a ornare odio, id bibendum est. Ut sem odio, vehicula laoreet neque eget, dictum malesuada ex. Nunc sed auctor turpis, et cursus elit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Integer gravida in turpis eu dictum. Cras eget elit nec erat tincidunt fringilla vitae ultrices ipsum. Nullam facilisis justo libero, ut euismod tortor facilisis non. Cras egestas ligula sed quam consectetur, ut faucibus nibh ornare. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Vestibulum lacinia lectus vitae nisl maximus, sit amet hendrerit odio sollicitudin. Integer eget ipsum eget libero tincidunt imperdiet. Donec cursus tortor velit, in pretium tortor rutrum sit amet. Phasellus bibendum augue eget dignissim molestie. Morbi tristique feugiat tempor. Vestibulum viverra justo quis luctus egestas. Donec pellentesque faucibus tortor, at faucibus augue fringilla ac. Nullam sed turpis turpis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Pellentesque ultricies, lorem ultrices placerat tincidunt, nisl dolor iaculis nunc, nec faucibus purus metus vel elit. Phasellus vel egestas purus. Curabitur ac ornare mauris, finibus commodo nulla. Nulla mi libero, porta nec consequat id, ullamcorper sed lacus. Phasellus auctor justo sed erat maximus sodales. Mauris mattis feugiat eros, at vehicula ipsum elementum a. Sed sit amet sapien consequat, congue massa a, sagittis justo. Quisque mauris sem, maximus sit amet ligula in, auctor maximus sem. Donec at metus arcu. Vestibulum in neque quis mauris porta cursus et ut felis. Pellentesque tempor lorem ipsum, a dapibus odio facilisis nec. Pellentesque pretium, augue in fringilla iaculis, tortor massa sagittis nunc, quis feugiat ligula odio ut libero. Maecenas pellentesque dapibus elementum. Mauris non lectus varius, mattis mi non, pharetra nunc. In elementum lectus ac convallis facilisis. In convallis rutrum mi, ac euismod leo iaculis sit amet. Fusce aliquet erat nisl, in molestie turpis pulvinar nec. Vivamus eget justo ullamcorper, molestie neque porta, convallis tortor. Nullam sed semper dui, nec auctor sem. Aliquam et vulputate ante. In hac habitasse platea dictumst. Aliquam eget ipsum placerat, molestie velit sit amet, sollicitudin quam. Integer interdum quis lacus non vulputate. Integer eget libero arcu. Donec vestibulum mollis nisi, non finibus arcu malesuada eu. Nullam faucibus purus sit amet lorem imperdiet, maximus ultrices dolor sodales. In mollis volutpat posuere. Vivamus non commodo velit. Ut ornare finibus lorem, vel dictum ante egestas et. Suspendisse convallis augue odio, nec mattis felis tristique nec. Pellentesque sit amet feugiat nisl. Morbi elementum fringilla enim suscipit imperdiet. Mauris ut erat id est fringilla aliquet nec sed odio. Nunc euismod, ex sit amet malesuada pulvinar, erat diam sollicitudin nulla, id ultrices nisl nulla eget nibh. Fusce mi enim, semper vel tristique in, feugiat vel nibh. Curabitur ultricies, ex quis ultrices consectetur, leo lorem dictum orci, eu lobortis nibh arcu fringilla massa. Quisque nibh lacus, imperdiet et neque et, elementum tincidunt diam. Nunc interdum scelerisque nunc eget bibendum. Quisque eu bibendum nisi, vel bibendum diam. Quisque vitae tincidunt leo. Aliquam ac dolor ut eros vehicula placerat eget eget nunc. Cras in consequat neque. Donec malesuada sed erat vel venenatis. Integer varius ante ante. Nam et tristique velit, quis tempus sem. Nunc sit amet ultrices magna. Pellentesque non quam porttitor, euismod arcu at, rutrum tortor. In nec nibh maximus, fringilla odio quis, auctor leo. Etiam ullamcorper est eget sapien sodales, vel molestie justo vestibulum. Duis vitae ex sapien. Fusce libero turpis, auctor a varius sit amet, fringilla vel mi. Duis metus enim, elementum vel leo nec, blandit vestibulum risus. Integer augue enim, vestibulum eu velit faucibus, convallis venenatis orci. In enim ex, scelerisque in mi sit amet, molestie posuere mi. Proin nec nisl lorem. Donec ut ligula faucibus, tristique felis id, pulvinar nisi. Nullam in magna ex. Quisque dignissim sem sit amet scelerisque pharetra. Maecenas accumsan, diam quis sollicitudin rutrum, diam orci ullamcorper ex, at tempus urna augue quis urna.Nullam auctor, elit ac venenatis feugiat, magna nunc porttitor ipsum, eget finibus lorem tellus sit amet dui. Praesent sem tellus, viverra eget efficitur efficitur, sodales a dolor. Suspendisse lobortis congue ante ac dictum. Phasellus porttitor mauris ipsum. Morbi in efficitur nibh. Fusce imperdiet eleifend dolor a auctor. Etiam elit dolor, molestie in porttitor non, bibendum vitae nibh. Nam libero turpis, malesuada et metus eget, tincidunt laoreet justo. Phasellus hendrerit orci in elit iaculis finibus. Nunc varius sapien id maximus condimentum. Ut at nulla sit amet libero pellentesque venenatis sed ut orci. Aliquam vitae lectus dapibus, sollicitudin ipsum eget, sollicitudin nibh."
# endregion


def replace_image(btn, img_bbox, img_mask):
    """Replace an image source of an image element with another image source."""
    fn = askopenfilename()
    img = salience.crop_img_to_mask(
        fn, img_mask, math.floor(height/32), drawFocal=False)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_bbox = np.multiply(img_bbox, math.floor(height/32))
    img_bbox = render.clamp_bbox(img_bbox, np.zeros((height, width)))

    # Place image
    img = ImageTk.PhotoImage(Image.fromarray(img))
    btn.configure(image=img)
    btn.image = img


def gen_bboxes(label):
    """Generate an editable layout conditioned on a text proportion label in [0,3]."""
    # Clear the frame
    for widgets in frm_render.winfo_children():
        widgets.destroy()

    # AI generation
    gen = generator.generate_layout(label)
    bboxes = refinement.extract_bboxes(gen)

    # region Add images
    # region Create binary mask (for image cropping)
    mask = np.ones((32, 24))
    for img_bbox in bboxes["image"]:
        render.fill_bbox(mask, img_bbox, 0)
    for text_bbox in bboxes["text"]:
        render.fill_bbox(mask, text_bbox, 1)
    for headline_bbox in bboxes["headline"]:
        render.fill_bbox(mask, headline_bbox, 1)
    # endregion

    # Crop images to the mask appropriately using image saliency
    counter = 0
    image_buttons = []
    image_masks = []
    for img_bbox in bboxes["image"]:
        img_mask = mask[img_bbox[2]:img_bbox[3], img_bbox[0]:img_bbox[1]]
        image_masks.append(img_mask)

        # Crop image
        img = salience.crop_img_to_mask(
            "./images/hong-nguyen-J4TEzb9mb3A-unsplash.jpg", img_mask, math.floor(height/32), drawFocal=False)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_bbox = np.multiply(img_bbox, math.floor(height/32))
        img_bbox = render.clamp_bbox(img_bbox, np.zeros((height, width)))

        # Place image
        img = ImageTk.PhotoImage(Image.fromarray(img))
        lbl_render = tk.Button(master=frm_render, image=img,
                               borderwidth=0, highlightthickness=0, command=lambda i=counter: replace_image(image_buttons[i], bboxes["image"][i], image_masks[i]))
        image_buttons.append(lbl_render)
        lbl_render.image = img
        lbl_render.place(x=img_bbox[0], y=img_bbox[2])

        counter += 1
    # endregion

    # region Add textboxes
    for text_bbox in bboxes["text"]:
        text_bbox = np.multiply(text_bbox, math.floor(height/32))
        lbl_render = tk.Text(master=frm_render, font=text_font,
                             borderwidth=0, highlightthickness=0)
        lbl_render.insert("1.0", lipsum)
        lbl_render.place(x=text_bbox[0], y=text_bbox[2], width=text_bbox[1] -
                         text_bbox[0], height=text_bbox[3]-text_bbox[2])
    # endregion

    # region Add headlines
    for headline_bbox in bboxes["headline"]:
        headline_bbox = np.multiply(headline_bbox, math.floor(height/32))
        lbl_render = tk.Text(
            master=frm_render, font=headline_font, borderwidth=0, highlightthickness=0)
        lbl_render.insert("1.0", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas quis elit vel risus elementum imperdiet nec vitae nunc. Ut fringilla lobortis massa, quis vehicula purus semper sed.")
        lbl_render.place(x=headline_bbox[0], y=headline_bbox[2], width=headline_bbox[1] -
                         headline_bbox[0], height=headline_bbox[3]-headline_bbox[2])
    # endregion


# region Tkinter
frm_main = tk.Frame(
    master=window, padx=20, pady=20)

# region Title
title = tk.Label(master=frm_main, font=h1,
                 text="Conditional Generative Modelling\nof Graphic Design Layouts")
title.pack(pady=(0, 20), side="top")
# endregion

# region Text proportion frame
frm_text_proportions = tk.Frame(master=frm_main)
btn_text_proportion_0 = tk.Button(
    master=frm_text_proportions,
    text="Tiny amount of text",
    command=lambda: gen_bboxes(0),
    relief=tk.FLAT,
    width=20,
    background="#E0E0E0",
    font=p
)
btn_text_proportion_1 = tk.Button(
    master=frm_text_proportions,
    text="Small amount of text",
    command=lambda: gen_bboxes(1),
    relief=tk.FLAT,
    width=20,
    background="#E0E0E0",
    font=p
)
btn_text_proportion_2 = tk.Button(
    master=frm_text_proportions,
    text="Medium amount of text",
    command=lambda: gen_bboxes(2),
    relief=tk.FLAT,
    width=20,
    background="#E0E0E0",
    font=p
)
btn_text_proportion_3 = tk.Button(
    master=frm_text_proportions,
    text="Large amount of text",
    command=lambda: gen_bboxes(3),
    relief=tk.FLAT,
    width=20,
    background="#E0E0E0",
    font=p
)

btn_text_proportion_0.pack(pady=(0, 10))
btn_text_proportion_1.pack(pady=(0, 10))
btn_text_proportion_2.pack(pady=(0, 10))
btn_text_proportion_3.pack()

frm_text_proportions.pack(side="top")
# endregion

# region Information
lbl_info = tk.Label(master=frm_main, font=p,
                    text="Click the text proportion buttons\nabove until you get a layout you like.\n\nNext, click on the images to replace\nthem with your own images.\n\nFinally, click on the textboxes, 'Ctrl-A'\nto select all the text, and then\nwrite whatever you would like.")
lbl_info.pack(side="bottom")
# endregion

# region Render frame
frm_render = tk.Frame(
    master=window, width=width, height=height, background="white", padx=10)
# endregion

frm_main.pack(side="left", fill="both")
frm_render.pack(side="left")

# Initialise with a "tiny text" layout on load
gen_bboxes(0)

window.mainloop()
# endregion
