import face_recognition

bill_image = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_encoding=face_recognition.face_encodings(bill_image)[0]

unknown_image=face_recognition.load_image_file('./img/unknown/bill-gates-4.jpg')
unknown_face_encoding=face_recognition.face_encodings(unknown_image)[0]

# compare faces
results=face_recognition.compare_faces([bill_encoding],unknown_face_encoding)
if results[0]:
     print('Match found')
else:
     print('No match')