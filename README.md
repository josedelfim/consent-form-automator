# Consent Form Automator

A local desktop tool (Python + Tkinter, built for macOS) that turns a pile of photographed, hand-signed **image-release forms** into clean, correctly named PDFs — each one with the right person's photo on it.

I built it for my own work as a video producer in Brazil. On a shoot, every person on camera signs a paper consent form. Afterwards someone has to match each form to the right face, crop the paper, paste a portrait onto it and file it under the person's name. With dozens of people per shoot, that took an afternoon. With this tool it takes about 30 minutes, most of it reviewing.

> Real batch: **92 faces** detected, **23 matches** suggested for review, **390 s** end to end.

## What it does

1. **Reads the people** from a WhatsApp export (photos with captions), a folder of named photos, or a folder of short videos where each person says their name.
   - Videos with a generic file name (`IMG_1037.mov`) are transcribed locally with **faster-whisper**, only the first seconds.
   - Faces are found with **YuNet** and compared with **SFace** (OpenCV), or Apple Vision on a Mac.
2. **Reads the forms** (JPG/PNG/PDF, local or from a public Google Drive folder), straightens and crops the paper, and runs **OCR only on the handwritten name field**, anchored on the printed sentence that precedes it. The rest of the form is ignored on purpose: clause text must never become a name.
3. **Matches forms to people as a whole batch**, not one by one: strong matches are reserved first, each person can only take one form, and a surname alone is never enough to match (`Ana Lima` will not grab `Carla Souza Lima`'s form).
4. **Sorts the result into OK / Review / No photo / Duplicates** and opens a review screen: big preview, zoom, manual crop, rotate, visual photo picker.
5. **Exports the PDFs**, named after the person, with the photo placed in a safe area of the page defined per form template (`modelos.json`). Forms without a match are exported with a `_SEMFOTO` suffix instead of guessing.

A second mode checks a finished video: it finds foreground faces with timestamps and suggests which signed forms cover the people who appear in it.

## Design choices

- **Safety over automation.** Low-confidence matches stay in *Review*; nothing is invented. When the handwriting is ambiguous the form keeps a generic name.
- **Everything runs locally.** Forms, photos and videos never go to an analysis service. Drive links are only used to download the files.
- **Speed by doing less.** Photos are opened at the size OCR needs, OCR reads one strip of the page, and videos that already have a person's name in the file name are not transcribed at all.
- **Duplicate check across batches.** A small local history flags people who already have a signed form from a previous shoot.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the two face models listed in models/README.md
python3 termos_app.py
```

Optional: a `nomes_brasileiros.json` file with ranked Brazilian first names and surnames improves name validation. It is not included (it comes from a third-party dataset); the app works without it, with a smaller built-in list.

The interface and the form templates are in Brazilian Portuguese, because the forms are.

## Status

Personal tool, used in real productions (v6.1, 2026). Published as a portfolio sample (~8,000 lines in a single file). Client names and real people's names were removed from this public version.
