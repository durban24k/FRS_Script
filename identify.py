import face_recognition
from PIL import Image, ImageDraw

bill_image=face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_encoding=face_recognition.face_encodings(bill_image)[0]

steve_image=face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steve_encoding=face_recognition.face_encodings(steve_image)[0]

# Create an array of encodings and names
known_face_encodings=[
     bill_encoding,
     steve_encoding
]
known_face_names=[
     "Bill Gates",
     "Steve Jobs"
]

# Load test image to find faces in
test_image=face_recognition.load_image_file('./img/groups/bill-steve.jpg')

# find the faces in test image
face_locations=face_recognition.face_locations(test_image)
face_encodings=face_recognition.face_encodings(test_image,face_locations)

# convert to PIL format
pil_image=Image.fromarray(test_image)

# Create a ImageDraw instance
draw= ImageDraw.Draw(pil_image)

# loop through the faces in test image
for(top, bottom, left, right), face_encoding in zip(face_locations,face_encodings):
     matches=face_recognition.compare_faces(known_face_encodings,face_encoding)

     name="Unknown Person"

     # if match
     if True in matches:
          first_match_index=matches.index(True)
          name=known_face_names[first_match_index]
     
     # Draw Box
     draw.rectangle(((left,top),(right,bottom)),outline=(255,0,0))

     # draw label
     text_width,text_height=draw.textsize(name)
     draw.rectangle(((left,bottom - text_height - 10),(right, bottom)),fill=(255,0,0),outline=(255,0,0))
     draw.text((left+6,bottom-text_height-5),name,fill=(0,0,0))

del draw

# Display image
pil_image.show()