# ChordLens web demo

A browser version of `realtime.py`, for presenting. Everything runs client-side:
the camera stream never leaves the machine.

```
webapp/
  public/              <- this folder is what gets deployed
    index.html         the whole app: UI, drawing, audio, prediction loop
    features.js        JavaScript port of src/features.py (the 85 features)
    model.json         exported logistic regression + scaler + voicings
  fixture.json         Python ground truth used to verify the port
  tools_verify_port.mjs  checks features.js still matches Python
```

## Run it locally

`getUserMedia` requires https **or** localhost, so opening `index.html` as a
`file://` URL will not work. Serve it:

```powershell
cd webapp\public
py -3.12 -m http.server 8080
```

Then open <http://localhost:8080>.

## Deploy to Vercel

Deploy **from `webapp/public`**, not from the repository root. The repo is about
2 GB because the image dataset is version-controlled, and Vercel's git
integration would clone all of it. The CLI uploads only this folder — about 48 KB.

```powershell
cd webapp\public
npx vercel --prod
```

Accept the defaults; when it asks for the project directory, it is already in it.

## After changing anything

If you retrain the model, re-export it, or the exported JSON goes stale:

```powershell
# from the repo root
.\.venv\Scripts\python.exe tools_export_model.py
node webapp\tools_verify_port.mjs
```

`tools_verify_port.mjs` recomputes the 85 features in JavaScript for five real
frames and compares them against the Python output stored in `fixture.json`. It
must print `PORT VERIFIED`. If it does not, the browser and the paper disagree
and the demo is reporting something the project did not measure.

## Known limitation

The published results (79.4% on unseen sessions, 68.9% on an unseen player) were
measured with MediaPipe's **Python** pipeline. This page uses MediaPipe's
**JavaScript** build, a different release, which may place landmarks slightly
differently. The classifier is bit-for-bit identical; the upstream landmark
detector is not. Live behaviour can therefore differ from the reported figures,
and the page says so.

`realtime.py` remains the reference demo and matches the paper exactly.
