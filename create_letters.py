from PIL import Image, ImageDraw, ImageFont
import os

alphabet = [chr(letter) for letter in range(ord('A'), ord('Z')+1)]
# fonts = ["arial.ttf", "calibri.ttf", "consola.ttf", "cour.ttf", "segoepr.ttf", "bahnschrift.ttf"]
font_dir = r'C:\Windows\Fonts'
exclude_fonts = ["holomdl2.ttf", "segmdl2.ttf", "marlett.ttf", "symbol.ttf", "webdings.ttf", "wingding.ttf"]
fonts = [filename for filename in os.listdir(font_dir) if filename.endswith('.ttf') and filename not in exclude_fonts]

for letter in alphabet:
    for font in fonts[:-4]:
        image = Image.new("L", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        
        sfont = ImageFont.truetype(font, 95)
        draw.text((30, 0), letter, fill="black", font=sfont)
        
        # imager = image.rotate(10, expand = True, fillcolor="white")
        # imagel = image.rotate(-10, expand = True, fillcolor="white")
        
        directory = f"train/{letter}"
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        image.save(f"{directory}/{font[:-4]}.png")
        # imager.save(f"{directory}/{font[:-4]}r.png")
        # imagel.save(f"{directory}/{font[:-4]}l.png")
        
    for font in fonts[-4:]:
        image = Image.new("L", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        
        sfont = ImageFont.truetype(font, 95)
        draw.text((30, 0), letter, fill="black", font=sfont)
        
        # imager = image.rotate(10, expand = True, fillcolor="white")
        # imagel = image.rotate(-10, expand = True, fillcolor="white")
        
        directory = f"test/{letter}"
            
        if not os.path.exists(directory):
            os.makedirs(directory)
                
        image.save(f"{directory}/{font[:-4]}.png")
        # imager.save(f"{directory}/{font[:-4]}r.png")
        # imagel.save(f"{directory}/{font[:-4]}l.png")