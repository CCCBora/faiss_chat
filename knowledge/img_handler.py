import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model_deplot = Pix2StructForConditionalGeneration.from_pretrained("google/deplot", torch_dtype=torch.bfloat16)
if device == "cuda":
    model_deplot = model_deplot.to(0)
processor_deplot = Pix2StructProcessor.from_pretrained("google/deplot")



def add_markup(table):
    try:
        parts = [p.strip() for p in table.splitlines(keepends=False)]
        if parts[0].startswith('TITLE'):
            result = f"Title: {parts[0].split(' | ')[1].strip()}\n"
            rows = parts[1:]
        else:
            result = ''
            rows = parts
        prefixes = ['Header: '] + [f'Row {i+1}: ' for i in range(len(rows) - 1)]
        return result + '\n'.join(prefix + row for prefix, row in zip(prefixes, rows))
    except:
        # just use the raw table if parsing fails
        return table

def process_image(image):
    inputs = processor_deplot(images=image, text="Generate the underlying data table for the figure below:",
                              return_tensors="pt").to(torch.bfloat16)
    if device == "cuda":
        inputs = inputs.to(0)
    predictions = model_deplot.generate(**inputs, max_new_tokens=512)
    table = processor_deplot.decode(predictions[0], skip_special_tokens=True).replace("<0x0A>", "\n")
    return table


if __name__ == "__main__":
    im = Image.open(r"meat-image.png")
    process_image(im)
