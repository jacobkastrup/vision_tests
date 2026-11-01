# vision_tests: parcel-label OCR with WeVisOne-VL

Use [WeVisOne-VL](https://github.com/Tencent/WeVisOne-VL) (Tencent, 2B / 4B, built on
Qwen3-VL) to read **a part of a parcel label that you describe in words**.

```bash
python read_label.py label.jpg "the tracking number under the barcode" "the recipient address block"
```
```
== the tracking number under the barcode  [crop, 1.4s]
1Z 999 AA1 01 2345 6784

== the recipient address block  [crop, 2.1s]
JANE DOE
12 HARBOUR ROAD
...
```

## Setup

```bash
./setup.sh          # clones Tencent/WeVisOne-VL into third_party/ and pip-installs it
```
Linux with an NVIDIA GPU is recommended. The 4B model needs about 10 GB VRAM in bf16;
use `--model tencent/WeVisOne-VL-2B` for less. Weights are downloaded from Hugging Face
on first run. Set `WEVISONE_MODEL=/path/to/checkpoint` to use a local copy.

## How it works

The model has no "read the text in region X" task. It has structured tasks: `referring`
(find a box from a text description), `ocr` (every word plus its box), and free-form
prompting. `parcel_ocr/reader.py` combines them into three strategies (`--mode`):

| mode | steps | notes |
|---|---|---|
| `crop` *(default)* | `referring` → pad and upscale the crop → `ocr` on the crop → join words in reading order | Small print is seen at high resolution. 2 model calls. |
| `filter` | `referring` + `ocr` on the whole label → keep words whose centre is inside the region | Text is read in full-label context. Can miss tiny text. |
| `ask` | a single free-form prompt: *"Read the {description}… reply with only the text"* | Experimental. The checkpoint is tuned for structured output. |
| `all` | runs all three | Use it to compare strategies on your labels. |

If the description matches nothing, the model answers `None`, which shows as `<not found>`.
When a description matches several boxes, they are merged by default (`--pick union`).
Use `--pick first` or `--pick largest` to take one box instead.

## Useful options

```bash
python read_label.py label.jpg "the parcel weight" --mode all --debug outputs/dbg.png   # overlay of regions + words
python read_label.py label.jpg "the sender postcode" --json      # boxes, raw model responses, latency
python read_label.py label.jpg "the tracking number" --rotate 90 # label photographed sideways
python read_label.py label.jpg "..." --backend vllm              # faster, if vllm is installed
```

## Batch evaluation

Describe your fields once in a JSON file (see `fields.example.json`):

```json
{ "tracking_number": "the tracking number printed under the large barcode",
  "recipient": "the recipient (ship to) name and address block" }
```

```bash
python batch_read.py labels/ fields.example.json -o outputs/results.csv --debug-dir outputs/debug
python batch_read.py labels/ fields.example.json -o outputs/results.csv --expected expected.csv
```
This writes a CSV with one column per field, plus a `.jsonl` file with boxes and raw responses.
With `--expected` (a CSV with an `image` column and the field columns) it also prints
exact-match accuracy per field, ignoring case and whitespace.

## Tips for descriptions

* Refer to what is printed or where it is: "the number under the large barcode", "the
  postcode in the ship-to block", "the text next to 'Weight:'".
* Keep each description to a single region. Run separate descriptions for separate fields.
* If a crop cuts off text, describe the whole block and post-process the result, or try `--mode filter`.

## Tests

```bash
python -m pytest -q        # parsing / reading-order tests, no GPU or model needed
```
