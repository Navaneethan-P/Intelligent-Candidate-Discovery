from PIL import Image

def remove_bg(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    data = img.getdata()
    
    new_data = []
    for item in data:
        if item[0] < 60 and item[1] < 60 and item[2] < 60:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    img.save(output_path, "PNG")

remove_bg(r"C:\Users\navan\.gemini\antigravity-ide\brain\2344d510-c281-4c4c-a453-52476b7ea20f\media__1781266006123.png", r"D:\AI Candidate Ranking System Implementation\ai_recruiter_submission\dashboard\logo.png")
