# vision_tests: parcel-label OCR with WeVisOne-VL vs PP-OCRv6

Read **a part of a parcel label that you describe in words**, and compare two OCR engines:

* [WeVisOne-VL](https://github.com/Tencent/WeVisOne-VL): Tencent's 2B/4B vision-language model,
  built on Qwen3-VL. It understands descriptions and can also do OCR. Needs a GPU.
* [PaddleOCR PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) (June 2026): a classic
  detection + recognition pipeline. Fast on CPU, but it cannot understand descriptions.

```bash
python read_label.py label.jpg "the tracking number under the barcode" --mode all --engine both
```
```text
# illustrative output
== the tracking number under the barcode  [crop/wevisone, 1.4s]
1Z 999 AA1 01 2345 6784
== the tracking number under the barcode  [crop/ppocr, 1.1s]
1Z 999 AA1 01 2345 6784
...
```

## How it works

Region finding always uses WeVisOne-VL's `referring` task (description → box). Reading is
then done by the engine you choose (`--engine wevisone|ppocr|both`):

| `--mode` | steps | engines |
|---|---|---|
| `crop` *(default)* | find the region → pad and upscale the crop → OCR the crop → join in reading order | both |
| `filter` | find the region; OCR the whole label; keep words/lines whose centre is in the region | both |
| `ask` | a single free-form prompt: *"Read the {description}… reply with only the text"* (experimental) | WeVisOne only |
| `full` | no region: OCR the whole label (baseline; the description is ignored) | both; `ppocr` needs no GPU |
| `all` | = `crop,filter,ask` | |

Modes can be combined: `--mode crop,filter,full`. A description that matches nothing gives
`<not found>`. If a description matches several boxes, they are merged (`--pick union`).
Use `--pick first` or `--pick largest` to take one. PP-OCRv6 returns **lines**, while
WeVisOne-VL returns **words**. In `filter` mode, a PP-OCR line that sticks out of the region
is kept or dropped as a whole.

## Running in the cloud (no local GPU needed)

The WeVisOne-VL 4B model needs a GPU with about 10 GB of memory. Three ways to rent one:

| option | best for | cost (approx.) | effort |
|---|---|---|---|
| **[Modal](https://modal.com)** (`cloud/modal_app.py`) | **recommended**: run scripts from your laptop; the GPU only runs while working | per second; L4 ≈ $0.80/h; free monthly credits | `pip install modal && modal setup` |
| **Google Colab** (`cloud/colab.ipynb`) | quick interactive tries in the browser | free T4 (limited), Pro ≈ $10/month for L4/A100 | open notebook, click run |
| **GPU VM** (RunPod, Lambda, Vast.ai, GCP, AWS) | long experiments, many images | ≈ $0.40–1.20/h while running, so **stop it when done** | SSH in, `git clone`, `./setup.sh` |

### Modal (recommended)

Your machine only needs Python with `modal` and `Pillow`. Images are uploaded, the models
run on a cloud GPU, and results are written locally.

```bash
pip install modal pillow
modal setup                                  # one-time browser login
modal run cloud/modal_app.py --images label.jpg --describe "the tracking number;the recipient address block"
modal run cloud/modal_app.py --images labels/ --fields fields.example.json \
    --mode crop,filter,full --engine both --expected expected.csv --out outputs/cloud --debug
```
The first run builds the container (about 10 minutes) and caches the weights in a Modal Volume.
Later runs start in under a minute. Use a different GPU with `MODAL_GPU=A10G` (or `A100-40GB`, `H100`)
and the smaller model with `--model tencent/WeVisOne-VL-2B`. The container stops itself
2 minutes after the last image.

### Colab

Open `cloud/colab.ipynb` in Colab (File → Open notebook → GitHub, or upload it), set the
runtime to GPU, and run the cells. On the free T4 the notebook uses `--dtype float16` automatically.

### Any GPU VM

```bash
git clone -b claude/adoring-dirac-2bbjtm https://github.com/jacobkastrup/vision_tests.git && cd vision_tests
./setup.sh
python batch_read.py labels/ fields.example.json --mode crop,filter,full --engine both
```

### Locally, CPU only

PP-OCRv6 runs on a laptop CPU. Only `--mode full --engine ppocr` works locally, because
reading a described region needs WeVisOne-VL.
```bash
./setup.sh --ppocr
python read_label.py label.jpg --mode full --engine ppocr
```

## Batch evaluation and engine comparison

Describe your fields once in a JSON file (see `fields.example.json`):
```json
{ "tracking_number": "the tracking number printed under the large barcode",
  "recipient": "the recipient (ship to) name and address block" }
```
```bash
python batch_read.py labels/ fields.example.json -o outputs/results.csv \
    --mode crop,filter,full --engine both --expected expected.csv --debug-dir outputs/debug
```
* `results.csv` has one row per (image, mode/engine) and one column per field. `results.jsonl` holds
  boxes, raw model output and latency.
* `expected.csv` (optional) has an `image` column plus one column per field with the correct text. It prints a table like this (illustrative numbers):
```
                    crop/ppocr     crop/wevisone      filter/ppocr   ...
  tracking_number        18/20             17/20             15/20
  sec/image               2.10              3.40              1.90
```
Matching ignores case and whitespace. For `full` runs it checks whether the expected text appears
anywhere in the full-label output.

## Other options

`--rotate 90` (sideways photo), `--json` (all details), `--debug overlay.png`, `--dtype float16` (T4/V100),
`--ppocr-tier medium|small|tiny`, `--ppocr-device gpu:0` (requires `paddlepaddle-gpu` instead of
`paddlepaddle`), `--backend vllm` (requires `pip install vllm==0.23.0`).

## Tips for descriptions

* Refer to what is printed or where it is: "the number under the large barcode", "the
  postcode in the ship-to block", "the text next to 'Weight:'".
* Keep each description to a single region. Run separate descriptions for separate fields.

## Tests

```bash
python -m pytest -q        # parsing / engine-conversion tests, no GPU or models needed
```
