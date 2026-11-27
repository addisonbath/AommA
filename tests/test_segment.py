import numpy as np

from aomma.stages.segment import _box_iou


def test_box_iou_compares_box_with_mask_bounds():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True
    assert _box_iou((2, 2, 6, 6), mask) == 1.0
    assert _box_iou((2, 2, 6, 10), mask) == 0.5
    assert _box_iou((7, 7, 9, 9), mask) == 0.0
    assert _box_iou((0, 0, 5, 5), np.zeros((10, 10), dtype=bool)) == 0.0


def test_box_fill_counts_smaller_things_in_front_but_not_bigger_ones():
    import numpy as np

    from aomma.stages.inventory import Inventory
    from aomma.stages.segment import MIN_FILL, box_fills

    def entry(name, box):
        return {"id": name, "bbox_2d": box, "description": "x", "size_m": [1, 1, 1], "on": "floor"}

    # A 100 x 100 photo. A shelf fills the lower half; books cover most of it; a "set" is boxed over the top half.
    inventory = Inventory.model_validate(
        {"objects": [entry("shelf", [0, 500, 1000, 1000]), entry("books", [100, 600, 900, 900]), entry("set", [0, 0, 1000, 500])]}
    )
    shelf = np.zeros((100, 100), dtype=bool)
    shelf[50:, :] = True
    books = np.zeros((100, 100), dtype=bool)
    books[60:90, 10:90] = True
    shelf &= ~books  # the shelf's own mask has a hole where the books are
    one_of_the_set = np.zeros((100, 100), dtype=bool)
    one_of_the_set[10:30, 10:40] = True  # the mask caught one small thing out of several

    fills = box_fills(inventory, {"shelf": shelf, "books": books, "set": one_of_the_set})
    assert fills["shelf"] == 1.0  # the hole is explained by the books in front
    assert fills["books"] == 1.0
    assert fills["set"] == 0.12 and fills["set"] < MIN_FILL


def test_a_second_look_splits_groups_or_keeps():
    import numpy as np

    from aomma.stages.inventory import Inventory
    from aomma.stages.segment import judge_second_look

    def entry(name, box):
        return {"id": name, "bbox_2d": box, "description": "x", "size_m": [1, 1, 1], "on": "floor"}

    def block(rows, cols):
        mask = np.zeros((100, 100), dtype=bool)
        mask[rows[0] : rows[1], cols[0] : cols[1]] = True
        return mask

    # A "set" boxed over the top half of a 100 x 100 photo, of which the mask caught one small thing (12% of the box).
    inventory = Inventory.model_validate({"objects": [entry("set", [0, 0, 1000, 500])]})
    masks = {"set": block((10, 30), (10, 40))}

    # Listed again as two things, each cleanly masked in its own box: split.
    apart = Inventory.model_validate({"objects": [entry("left", [100, 100, 400, 300]), entry("right", [500, 100, 900, 400])]})
    assert judge_second_look(inventory, masks, "set", apart, {"left": block((10, 30), (10, 40)), "right": block((10, 40), (50, 90))})[0] == "split"

    # Two things whose masks are scraps of their boxes: they stay one object, under everything that was found.
    tangled = Inventory.model_validate({"objects": [entry("left", [0, 0, 600, 500]), entry("right", [400, 0, 1000, 500])]})
    done, together = judge_second_look(inventory, masks, "set", tangled, {"left": block((0, 20), (0, 25)), "right": block((30, 50), (70, 100))})
    assert done == "grouped" and together.sum() > masks["set"].sum()

    # A second look that finds nothing new changes nothing.
    same = Inventory.model_validate({"objects": [entry("set_again", [0, 0, 1000, 500])]})
    assert judge_second_look(inventory, masks, "set", same, {"set_again": block((10, 30), (10, 40))})[0] == "kept"
    # One thing with a tighter box that its mask now fills: take the redrawn entry.
    tight = Inventory.model_validate({"objects": [entry("thing", [100, 100, 400, 300])]})
    assert judge_second_look(inventory, masks, "set", tight, {"thing": block((10, 30), (10, 40))})[0] == "redrawn"


def test_an_object_with_nothing_found_does_not_take_the_others_masks_down():
    import io

    from PIL import Image

    from aomma.backends.comfy import ComfyError
    from aomma.stages.segment import _run

    size = (40, 30)
    boxes = {"shelf": (0, 0, 20, 30), "book": (20, 10, 30, 20), "ghost": (30, 0, 40, 10)}
    found = {"shelf": boxes["shelf"], "book": boxes["book"]}  # nothing matches "ghost"

    def picture(box):
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        if box:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 255
        buffer = io.BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        return buffer.getvalue()

    class Server:
        """Answers mask jobs the way ComfyUI does: a batch holds whatever was found, after anything it is batched behind."""

        def __init__(self, fails_with_ghost: bool):
            self.fails_with_ghost, self.jobs = fails_with_ghost, []

        def run(self, workflow, label=""):
            ids = [key[4:] for key in workflow if key.startswith("out_")]
            self.jobs.append(ids)
            if self.fails_with_ghost and "ghost" in ids:
                raise ComfyError("workflow failed: index 0 is out of bounds for dimension 0 with size 0")
            outputs = {}
            for oid in ids:
                batch = workflow[f"out_{oid}"]["inputs"]["images"][0]
                # Without a blank in front, an empty batch is what makes the real server fail.
                assert workflow[batch]["class_type"] == "ImageBatch" and workflow[batch]["inputs"]["image1"] == ["blank", 0]
                outputs[f"out_{oid}"] = {"images": [{"filename": "blank"}] + ([{"filename": oid}] if oid in found else [])}
            return outputs

        def fetch(self, file):
            return picture(found.get(file["filename"]))

    # The blank in front keeps every batch non-empty: one job, and the ghost simply has no mask.
    server = Server(fails_with_ghost=False)
    masks, infos = _run(server, "photo.jpg", size, boxes, "box", list(boxes), {})
    assert server.jobs == [["shelf", "book", "ghost"]]
    assert infos["shelf"].box_iou == 1.0 and infos["book"].box_iou == 1.0
    assert not masks["ghost"].any() and infos["ghost"].area_px == 0

    # If the job fails anyway, each object is tried alone and only the one that fails goes without.
    server = Server(fails_with_ghost=True)
    masks, infos = _run(server, "photo.jpg", size, boxes, "box", list(boxes), {})
    assert server.jobs == [["shelf", "book", "ghost"], ["shelf"], ["book"], ["ghost"]]
    assert infos["shelf"].box_iou == 1.0 and infos["book"].box_iou == 1.0
    assert not masks["ghost"].any() and infos["ghost"].box_iou == 0.0
