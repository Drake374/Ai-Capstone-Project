import glob
from enroll import enroll_from_images

student_id = "stu_demo_1"

# pick a person who has many images in LFW (change the folder name if needed)
imgs = glob.glob(r"D:\centennial_4thsem\AIcapstoneproject\codefiles\capstone\lfwfunneled\lfw_funneled\George_W_Bush\*.jpg")[:10]

out = enroll_from_images(student_id, imgs)
print("Saved embedding:", out)
