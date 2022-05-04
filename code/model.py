import generator
import refinement
import render
import cv2 as cv


def ours(label):
    """Generate and render a layout using our best techniques."""
    gen = generator.generate_layout(label)
    bboxes = refinement.extract_bboxes(gen)
    return render.render(bboxes)


def b1(label):
    """Generate and render a layout using our best techniques, but without our refinement step."""
    gen = generator.generate_layout(label)
    bboxes = refinement.extract_bboxes_b1(gen)
    return render.render(bboxes)


def b2(label):
    """Generate and render a layout using our best techniques, but without our image cropping step."""
    gen = generator.generate_layout(label)
    bboxes = refinement.extract_bboxes(gen)
    return render.render_b2(bboxes)


# Examples of how to run
cv.imwrite("render0.png", ours(0))
cv.imwrite("render1.png", ours(1))
cv.imwrite("render2.png", ours(2))
cv.imwrite("render3.png", ours(3))
