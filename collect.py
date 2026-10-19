import argparse
import json
import os

import cv2
import gradio as gr


# number of clicks allowed per image
MIN_CLICKS = 3
MAX_CLICKS = 6

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp") # accepted image formats


def list_images(images_dir):
    # sorted list of the images in the folder
    if not os.path.isdir(images_dir):
        return []
    return sorted(f for f in os.listdir(images_dir)
                  if f.lower().endswith(IMAGE_EXTENSIONS))


def load_data(out_path):
    # load the clicks already saved, empty dict if there is no file yet
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {}


def save_data(data, out_path):
    # write to a temporary file first, so the json is never left half written
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, out_path)


def draw_clicks(image_rgb, clicks):
    # draw a numbered circle on every click
    annotated = image_rgb.copy()
    for i, (x, y) in enumerate(clicks, start=1):
        cv2.circle(annotated, (x, y), 14, (0, 0, 0), -1) # black border
        cv2.circle(annotated, (x, y), 12, (255, 80, 40), -1) # orange circle
        cv2.putText(annotated, str(i), (x - 6, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA) # click number
    return annotated


# state of the collection: images, saved clicks and current image
class Session:
    def __init__(self, images_dir, out_path):
        self.images_dir = images_dir
        self.out_path = out_path
        self.images = list_images(images_dir)
        self.data = load_data(out_path) # clicks already saved
        self.index = 0 # index of the current image
        self.clicks = [] # clicks on the current image
        self.redoing = False # true when redoing a single image
        self._skip_done()

    def _skip_done(self):
        # move to the first image without clicks
        while self.index < len(self.images) and self.images[self.index] in self.data:
            self.index += 1

    @property
    def finished(self):
        # true when all images are done (and we are not redoing one)
        return self.index >= len(self.images) and not self.redoing

    @property
    def name(self):
        # filename of the current image, None if finished
        if self.redoing:
            return self.images[self.index]
        return None if self.finished else self.images[self.index]

    def current_image(self):
        # load the current image
        if self.name is None:
            return None
        path = os.path.join(self.images_dir, self.name)
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None # opencv loads BGR, gradio wants RGB

    def add_click(self, x, y):
        # add a click only if the maximum is not reached
        if len(self.clicks) < MAX_CLICKS:
            self.clicks.append((int(x), int(y)))

    def undo(self):
        # remove the last click
        if self.clicks:
            self.clicks.pop()

    def annotated_images(self):
        # images that already have clicks
        return [img for img in self.images if img in self.data]

    def redo(self, image_name):
        # go to the chosen image and start its clicks from scratch
        if image_name not in self.images:
            return
        self.index = self.images.index(image_name)
        self.clicks = []
        self.redoing = True

    def reset_all(self):
        # delete all the clicks and restart from the first image
        self.data = {}
        save_data(self.data, self.out_path)
        self.index = 0
        self.clicks = []
        self.redoing = False

    def save_and_next(self):
        # error message if there are not enough clicks
        if len(self.clicks) < MIN_CLICKS:
            return False, f"At least {MIN_CLICKS} clicks are required (you have {len(self.clicks)})."
        # save the clicks with their order
        self.data[self.name] = [
            {"x": x, "y": y, "order": i + 1}
            for i, (x, y) in enumerate(self.clicks)
        ]
        save_data(self.data, self.out_path)
        self.clicks = []
        if self.redoing:
            # after a redo go back to the first image without clicks
            self.redoing = False
            self.index = 0
            self._skip_done()
        else:
            self.index += 1
            self._skip_done()
        return True, "Saved."

    def status(self):
        # text with the progress of the collection
        if self.redoing:
            return (f"Redoing: {self.name} --- "
                    f"clicks: {len(self.clicks)}/{MAX_CLICKS} (minimum {MIN_CLICKS})")
        if self.finished:
            return f"Done. {len(self.data)} images annotated in total."
        return (f"{self.index + 1}/{len(self.images)} --- {self.name} --- "
                f"clicks: {len(self.clicks)}/{MAX_CLICKS} (minimum {MIN_CLICKS})")


def _home_msg(session, out_path):
    # text of the home panel
    total = len(session.images)
    done = len(session.annotated_images())
    remaining = total - done

    lines = ["## 📋 Click Collection --- Control Panel\n"]
    lines.append(f"**{done}/{total}** images annotated.")
    if remaining > 0:
        lines.append(f" **{remaining}** images remaining.\n")
    else:
        lines.append(f" ✅ All images have been annotated!\n")
    lines.append(f"\nData is saved to `{out_path}`.\n")
    return "".join(lines)


_ALL_KEYS = 11 # number of ui components updated by the two functions below

def _home_outputs(session, out_path):
    # show the home panel and hide the editing view
    annotated = session.annotated_images()
    has_remaining = not session.finished
    return (
        gr.update(value=None, visible=False), # image
        gr.update(visible=True), # home_panel
        gr.update(value=_home_msg(session, out_path)), # home_msg
        gr.update(visible=has_remaining,
                  value="▶️ Continue collection" if annotated else "▶️ Start collection"), # home_continue_btn
        gr.update(choices=session.images,
                  value=session.images[0] if session.images else None,
                  visible=True), # redo_dd
        gr.update(visible=True), # home_redo_btn
        gr.update(visible=bool(annotated)), # home_reset_btn
        gr.update(visible=False), # edit_row
        gr.update(visible=False), # undo_btn
        gr.update(visible=False), # next_btn
        gr.update(visible=False), # stop_btn
    )

def _editing_outputs(session, show_stop=True):
    # show the editing view and hide the home panel, show_stop=False during a redo
    img = session.current_image()
    return (
        gr.update(value=img, visible=True), # image
        gr.update(visible=False), # home_panel
        gr.update(), # home_msg
        gr.update(), # home_continue_btn
        gr.update(), # redo_dd
        gr.update(), # home_redo_btn
        gr.update(), # home_reset_btn
        gr.update(visible=True), # edit_row
        gr.update(visible=True), # undo_btn
        gr.update(visible=True), # next_btn
        gr.update(visible=show_stop), # stop_btn
    )


def build_interface(images_dir, out_path):
    # build the gradio interface
    session = Session(images_dir, out_path)
    annotated = session.annotated_images()

    with gr.Blocks(title="NOVA --- Click Collection") as demo:
        gr.Markdown(
            "# Click Collection\n"
            "Click on the image in order of decreasing importance "
            f"(from most to least important), {MIN_CLICKS}-{MAX_CLICKS} clicks, then **Next**."
        )
        status = gr.Textbox(visible=False) # hidden, only used as second output

        # home panel
        with gr.Column(visible=True) as home_panel:
            home_msg = gr.Markdown(value=_home_msg(session, out_path))
            continue_label = "▶️ Continue collection" if annotated else "▶️ Start collection"
            home_continue_btn = gr.Button(
                continue_label, variant="primary",
                visible=not session.finished
            )
            with gr.Row():
                redo_dd = gr.Dropdown(
                    label="Image to annotate",
                    choices=session.images,
                    value=session.images[0] if session.images else None,
                    interactive=True,
                ) # choose a single image to annotate
                home_redo_btn = gr.Button("🔄 Do single", variant="secondary")
            home_reset_btn = gr.Button("🗑️ Redo all", variant="stop",
                                        visible=bool(annotated)) # only if there are saved clicks

        # editing view, hidden until the collection starts
        image = gr.Image(type="numpy", interactive=False, height=500,
                         visible=False)
        with gr.Row(visible=False) as edit_row:
            undo_btn = gr.Button("Undo last click")
            next_btn = gr.Button("Next", variant="primary")
            stop_btn = gr.Button("⏹ Save and exit", variant="stop")

        # same order as _home_outputs and _editing_outputs
        all_outputs = [
            image, home_panel, home_msg, home_continue_btn,
            redo_dd, home_redo_btn, home_reset_btn,
            edit_row, undo_btn, next_btn, stop_btn,
        ]

        def on_continue():
            # start or continue the collection
            if session.finished:
                return _home_outputs(session, out_path)
            return _editing_outputs(session)

        def on_redo(selected_image):
            # redo the image chosen in the dropdown
            if not selected_image:
                return tuple(gr.update() for _ in range(_ALL_KEYS))
            session.redo(selected_image)
            return _editing_outputs(session, show_stop=False)

        def on_reset():
            # delete everything and restart
            session.reset_all()
            return _editing_outputs(session)

        def on_click(evt: gr.SelectData):
            # save the click and redraw the image
            if session.finished or session.current_image() is None:
                return gr.update(), gr.update()
            session.add_click(evt.index[0], evt.index[1])
            return draw_clicks(session.current_image(), session.clicks), gr.update()

        def on_undo():
            # remove the last click and redraw the image
            session.undo()
            img = session.current_image()
            if img is None:
                return gr.update(), gr.update()
            return (draw_clicks(img, session.clicks) if session.clicks else img,
                    gr.update()) # clean image if there are no clicks left

        def on_next():
            # save the clicks and go to the next image
            was_redoing = session.redoing
            ok, msg = session.save_and_next()
            if ok and (session.finished or was_redoing):
                return _home_outputs(session, out_path) # back home when finished or after a redo
            if not ok:
                # not enough clicks, stay on the same image
                img = session.current_image()
                return (
                    gr.update(value=draw_clicks(img, session.clicks) if session.clicks else img),
                    *[gr.update() for _ in range(_ALL_KEYS - 1)],
                )
            return _editing_outputs(session)

        def on_stop():
            # save only if there are enough clicks, then back home
            if len(session.clicks) >= MIN_CLICKS:
                session.save_and_next()
            return _home_outputs(session, out_path)

        # connect buttons and image to the callbacks
        home_continue_btn.click(on_continue, outputs=all_outputs)
        home_redo_btn.click(on_redo, inputs=[redo_dd], outputs=all_outputs)
        home_reset_btn.click(on_reset, outputs=all_outputs)

        image.select(on_click, outputs=[image, status])
        undo_btn.click(on_undo, outputs=[image, status])
        next_btn.click(on_next, outputs=all_outputs)
        stop_btn.click(on_stop, outputs=all_outputs)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click collection for NOVA.")
    parser.add_argument("--images", default="img") # folder with the images
    parser.add_argument("--out", default="clicks.json") # file where the clicks are saved
    args = parser.parse_args()

    # error message if the folder has no images
    if not list_images(args.images):
        raise SystemExit(f"No images found in '{args.images}'.")

    build_interface(args.images, args.out).launch()
