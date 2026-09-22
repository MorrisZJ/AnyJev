"""The image path, end to end through the real prompt path with a synthetic
model that never decodes a pixel."""
import pytest
from PIL import Image as PILImage

from anyjev import Decider, Image, Question
from anyjev.backends.fake import FakeBackend
from anyjev.media import blank_image
from anyjev.readout import build_prompt, prompt_images, user_content
from anyjev.state import content_free_state, render_state, split_state

RED = PILImage.new("RGB", (8, 8), (255, 0, 0))
GREEN = PILImage.new("RGB", (8, 8), (0, 255, 0))
BLUE = PILImage.new("RGB", (8, 8), (0, 0, 255))

OPTIONS = ["red", "green", "blue", "grey"]
COLOUR = {Image(RED).key: "red", Image(GREEN).key: "green", Image(BLUE).key: "blue"}


def no_text_signal(state, option):
    return 0.0


def sees_colour(image_keys, option):
    """The synthetic model can only read the picture."""
    return 3.0 if image_keys and COLOUR.get(image_keys[0]) == option else 0.0


def vlm(**kw) -> FakeBackend:
    return FakeBackend(no_text_signal, image_content=sees_colour, **kw)


# ---- state ------------------------------------------------------------

def test_text_only_states_render_exactly_as_before():
    assert render_state(None) == ""
    assert render_state("plain") == "plain"
    assert render_state({"a": 1}) == '{\n  "a": 1\n}'
    assert render_state([{"role": "user", "content": "hi"}]) == "user: hi"


def test_split_state_pulls_pictures_out_and_leaves_a_marker():
    text, images = split_state({"screenshot": RED, "note": "hello"})
    assert len(images) == 1 and images[0].key == Image(RED).key
    assert "<image 1>" in text and "hello" in text


def test_marker_order_matches_image_order():
    state = [{"role": "user", "content": [
        {"type": "image", "image": "/tmp/a.png"},
        {"type": "text", "text": "and then"},
        {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}},
    ]}]
    text, images = split_state(state)
    assert [im.kind for im in images] == ["path", "url"]
    assert text.index("<image 1>") < text.index("and then") < text.index("<image 2>")


def test_a_state_with_no_pictures_has_no_images():
    assert split_state({"a": 1})[1] == ()


# ---- media ------------------------------------------------------------

def test_key_is_content_addressed_for_pixels_and_source_addressed_for_paths():
    assert Image(RED).key == Image(RED.copy()).key
    assert Image(RED).key != Image(GREEN).key
    assert Image("/tmp/a.png").key == Image("/tmp/a.png").key != Image("/tmp/b.png").key
    assert Image("https://example.com/a.png").kind == "url"
    assert Image(Image(RED)).key == Image(RED).key


def test_load_decodes_path_and_bytes_to_rgb(tmp_path):
    path = tmp_path / "red.png"
    RED.save(path)
    from_path = Image(str(path)).load()
    from_bytes = Image(path.read_bytes()).load()
    assert from_path.mode == from_bytes.mode == "RGB"
    assert from_path.size == (8, 8) and from_path.getpixel((0, 0)) == (255, 0, 0)


def test_blank_probe_image_is_flat_grey_and_stable():
    assert blank_image().key == blank_image().key
    assert blank_image().load().getpixel((0, 0)) == (127, 127, 127)


# ---- readout ----------------------------------------------------------

def test_user_turn_interleaves_the_picture_where_the_marker_sat():
    q = Question.choice("What colour?", OPTIONS)
    spec = build_prompt("<image 1>\na caption", q, [0, 1, 2, 3], images=(Image(RED),))
    parts = user_content(spec)
    assert parts[0] == {"type": "text", "text": "State:\n"}
    assert parts[1] == {"type": "image"}
    assert "a caption" in parts[2]["text"] and "What colour?" in parts[2]["text"]


def test_user_turn_is_a_plain_string_without_pictures():
    q = Question.choice("What colour?", OPTIONS)
    assert isinstance(user_content(build_prompt("text", q, [0, 1, 2, 3])), str)


def test_a_literal_marker_in_the_question_is_not_a_second_picture():
    # MMMU-style question text refers to "<image 1>" itself
    q = Question.choice("What colour is <image 1>?", OPTIONS)
    text, images = split_state(RED)
    spec = build_prompt(text, q, [0, 1, 2, 3], images=images)
    assert sum(p == {"type": "image"} for p in user_content(spec)) == 1
    assert [im.key for im in prompt_images(spec)] == [Image(RED).key]


def test_images_reach_the_backend_in_placeholder_order_whatever_the_text_says():
    # the user's own note mentions "<image 2>" before either real marker: the
    # placeholders and the image list must still line up one to one
    state = {"note": "compare with <image 2>", "a": RED, "b": BLUE}
    text, images = split_state(state)
    q = Question.choice("What colour?", OPTIONS)
    spec = build_prompt(text, q, [0, 1, 2, 3], images=images)
    placeholders = sum(p == {"type": "image"} for p in user_content(spec))
    assert placeholders == len(prompt_images(spec)) == 2
    assert {im.key for im in prompt_images(spec)} == {Image(RED).key, Image(BLUE).key}
    be = vlm()
    Decider(be).decide(state, [q], level="raw")
    assert all(seen == prompt_images(spec) for seen in be.images_seen)


def test_path_images_do_not_hold_file_handles(tmp_path):
    import os

    path = tmp_path / "red.png"
    RED.save(path)
    before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None
    refs = [Image(str(path)) for _ in range(50)]
    for ref in refs:
        ref._key = ref._key + str(id(ref))        # 50 distinct refs to the same file
        ref.load()
    if before is not None:
        assert len(os.listdir("/proc/self/fd")) - before < 5


# ---- decider ----------------------------------------------------------

def test_pictures_reach_the_backend_once_per_distinct_prompt():
    be = vlm()
    q = Question.choice("What colour?", OPTIONS, name="colour")
    Decider(be).decide_batch([Image(RED), Image(RED)], q)
    assert be.prompts_seen == 4                       # 4 permutations, both states deduplicated
    assert all(len(imgs) == 1 and imgs[0].key == Image(RED).key for imgs in be.images_seen)


def test_two_different_pictures_are_two_prompts_each():
    be = vlm()
    q = Question.choice("What colour?", OPTIONS, name="colour")
    Decider(be).decide_batch([Image(RED), Image(BLUE)], q)
    assert be.prompts_seen == 8
    assert {imgs[0].key for imgs in be.images_seen} == {Image(RED).key, Image(BLUE).key}


def test_a_text_only_backend_refuses_a_state_with_pictures():
    d = Decider(FakeBackend(no_text_signal))
    with pytest.raises(ValueError, match="text-only"):
        d.decide(Image(RED), [Question.choice("What colour?", OPTIONS)])


def test_raw_is_fooled_by_position_bias_on_a_picture_l0_is_not():
    be = vlm(position_bias=[4.0, 0, 0, 0])            # loves position A
    q = Question.choice("What colour?", OPTIONS, name="colour")
    d = Decider(be)
    raw = d.decide(Image(GREEN), [q], level="raw")["colour"]
    l0 = d.decide(Image(GREEN), [q], level="L0")["colour"]
    assert raw.argmax == "red"                        # wrong: whatever sits at position A
    assert l0.argmax == "green" and l0.level == "L0"
    assert l0.diagnostics["n_images"] == 1


def test_the_content_free_probe_blanks_the_picture_too():
    be = FakeBackend(no_text_signal, image_content=sees_colour, label_prior={"Yes": 2.0})
    q = Question.noul("Is it warm?", name="warm")
    d = Decider(be, prior="content_free")
    raw = d.decide(Image(RED), [q], level="raw")["warm"]
    l0 = d.decide(Image(RED), [q], level="L0")["warm"]
    assert raw.p_true > 0.85                          # the model just likes saying Yes
    assert abs(l0.p_true - 0.5) < 1e-6                # divided out exactly
    assert l0.diagnostics["prior_method"] == "content_free"
    probe_key = blank_image().key
    assert any(imgs and imgs[0].key == probe_key for imgs in be.images_seen)


def test_probes_carry_as_many_blanks_as_the_state_carries_pictures():
    text, images = content_free_state("N/A", 2)
    assert text == "<image 1>\n<image 2>\nN/A"
    assert len(images) == 2 and images[0].key == blank_image().key
    assert content_free_state("", 0) == ("", ())


def test_probes_are_shared_across_states_with_the_same_picture_count():
    be = vlm()
    q = Question.choice("What colour?", OPTIONS)
    d = Decider(be, prior="content_free")
    d.decide_batch([Image(RED), Image(GREEN), Image(BLUE)], q)
    # 3 states x 4 perms, then 4 perms x 3 probes shared once in their own call = 24
    assert be.calls == 2 and be.prompts_seen == 24
    d.decide(Image(RED), [q])
    assert be.calls == 3 and be.prompts_seen == 28    # cf prior cached, no probes reissued


def test_picture_prompts_never_take_the_shared_prefix_path():
    # the shared-prefix contract carries text only; a picture prompt must go flat with its images
    q = Question.choice("What colour?", OPTIONS)
    be = vlm()
    Decider(be, shared_prefix=True).decide_batch([Image(RED), Image(BLUE)], q)
    assert be.shared_calls == 0 and len(be.images_seen) == 8
    text_be = FakeBackend(no_text_signal)
    Decider(text_be, shared_prefix=True).decide_batch(["a", "b"], q)
    assert text_be.shared_calls == 1                  # text states still share


def test_adaptive_shifts_carry_pictures_and_a_probe_per_picture_count():
    be = vlm(position_bias=[4.0, 0, 0, 0])
    q = Question.choice("What colour?", OPTIONS, name="colour")
    d = Decider(be, prior="content_free", adaptive_shifts=True)
    two = {"a": GREEN, "b": GREEN}                    # two pictures: its own probe
    decs = d.decide_batch([Image(GREEN), two], q)
    assert [x.argmax for x in decs] == ["green", "green"]
    assert [x.diagnostics["n_images"] for x in decs] == [1, 2]
    assert decs[0].diagnostics["adaptive"] and decs[0].diagnostics["prior_method"] == "content_free"
    assert {k[1] for k in d._cf_cache} == {1, 2}
    blank = blank_image().key
    assert any(len(s) == 2 and all(im.key == blank for im in s) for s in be.images_seen)


def test_l1_freezes_the_prior_on_picture_states():
    be = vlm(label_prior={"A": 1.5})
    q = Question.choice("What colour?", OPTIONS, name="colour")
    d = Decider(be)
    states = [Image(RED), Image(GREEN), Image(BLUE)] * 10
    art = d.calibrate(q, states, [OPTIONS.index(COLOUR[Image(s).key]) for s in states])
    assert art.get("prior") is not None               # the batch prior from the calibration set
    r = d.decide(Image(GREEN), [q], level="L1")["colour"]
    assert r.level == "L1" and r.argmax == "green"
    assert r.diagnostics["prior_method"].startswith("frozen:")


def test_l1_calibration_works_on_picture_states():
    be = vlm()
    q = Question.choice("What colour?", OPTIONS, name="colour")
    d = Decider(be)
    states = [Image(RED), Image(GREEN), Image(BLUE)] * 20
    labels = [OPTIONS.index(COLOUR[Image(s).key]) for s in states]
    art = d.calibrate(q, states, labels)
    assert art["temperature"] > 0 and art["question"] == q.key
    l0 = d.decide(Image(BLUE), [q], level="L0")["colour"]
    l1 = d.decide(Image(BLUE), [q], level="L1")["colour"]
    assert l1.level == "L1" and l1.argmax == l0.argmax == "blue"
    # this synthetic model is right every time, so matching confidence to
    # accuracy means sharpening, not softening
    assert l1.confidence > l0.confidence
