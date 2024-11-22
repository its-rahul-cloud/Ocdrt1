import torch
import timm
import model
from transformers import RobertaModel

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

class DynRT(torch.nn.Module):
  # define model elements
    def __init__(self,bertl_text,vit, opt):
        super(DynRT, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.vit=vit
        if not self.opt["finetune"]:
            freeze_layers(self.bertl_text)
            freeze_layers(self.vit)
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        self.trar = model.TRAR.DynRT(opt)
        self.sigm = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(opt["output_size"],2)
        )

    def vit_forward(self,x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x[:,1:]

    # forward propagate input
    def forward(self, input):
        # (bs, max_len, dim)
        bert_embed_text = self.bertl_text.embeddings(input_ids = input[self.input1])
        # (bs, max_len, dim)
        # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        for i in range(self.opt["roberta_layer"]):
            bert_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
            bert_embed_text = bert_text
        # (bs, grid_num, dim)
        img_feat = self.vit_forward(input[self.input2])

        (out1, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text,input[self.input3].unsqueeze(1).unsqueeze(2))

        out = self.classifier(out1)
        result = self.sigm(out)

        del bert_embed_text, bert_text, img_feat, out1, out
    
        return result, lang_emb, img_emb

def build_DynRT(opt,requirements):

    
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    if "vitmodel" not in opt:
        opt["vitmodel"]="vit_base_patch32_224"
    vit = timm.create_model(opt["vitmodel"], pretrained=True)
    return DynRT(bertl_text,vit,opt)

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

def process_single_image(image_path):
    """
    Processes a single image for OCR using the TrOCR model.

    Parameters:
        image_path (str): File path of the image to process.

    Returns:
        tuple:
            - bool: True if OCR is successful and text length >= 5, False otherwise.
            - str: The generated OCR text if successful, otherwise an error message or empty string.
    """
    try:
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        # Load the TrOCR processor and model
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        # Load the image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return False, "Error loading image."

        # Preprocess the image
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Generate OCR text
        generated_ids = model.generate(pixel_values)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

        # Determine success based on text length
        if len(generated_text) < 5:
            return False, generated_text
        else:
            return True, generated_text

    except Exception as e:
        return False, " "




# Example usage
image_url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
text = process_single_image(image_url)
#print("Extracted Text:", text)

#Verbal irony is a figure of speech that communicates the opposite of what is said, 
# while sarcasm is a form of irony that is directed at a person, with the intent to criticise.
# Function to check for irony
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import urllib.request
import csv

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# Function to check for irony in a single text pair
def detect_irony(text_pair):
    task = 'irony'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Download label mapping
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    def analyze_text(text):
        if len(text.split()) <= 4:
            return 0

        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)[::-1]
        return scores[ranking[0]]  # Return the highest score

    # Analyze the texts in the pair
    score1 = analyze_text(text_pair[0])
    score2 = analyze_text(text_pair[1])

    return 1 if score1 > 0.7 or score2 > 0.7 else 0




# Example usage with images
def process_image_pair(text1, text2):

    if not text1.strip() or not text2.strip():
        print("One or both images do not contain text.")
        return None

    result = detect_irony((text1, text2))
    return result


# Test with image pairs
image1_path = "image1.jpg"  # Replace with the path to your first image
image2_path = "image2.jpg"  # Replace with the path to your second image

result = process_image_pair(image1_path, image2_path)

if result is not None:
    print(f"Result for image pair: {result}")
