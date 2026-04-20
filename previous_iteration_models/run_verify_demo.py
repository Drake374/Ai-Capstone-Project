from verify import verify_image

img_same = r"D:\centennial_4thsem\AIcapstoneproject\codefiles\capstone\lfwfunneled\lfw_funneled\George_W_Bush\George_W_Bush_0001.jpg"
img_diff = r"D:\centennial_4thsem\AIcapstoneproject\codefiles\capstone\lfwfunneled\lfw_funneled\Tony_Blair\Tony_Blair_0001.jpg"

print("SAME:", verify_image("stu_demo_1", img_same))
print("DIFF:", verify_image("stu_demo_1", img_diff))
