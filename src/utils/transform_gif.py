from PIL import Image
import glob
import os

def transform_gif(dir, frame_duration=1200):
    # GIF 파일의 이름과 프레임 속도를 설정합니다.
    gif_name = os.path.join(dir,"output.gif")

    png_files = sorted(glob.glob(os.path.join(dir, "image_logs", "*.png")))
    # 첫 번째 PNG 파일을 열어 GIF 파일의 기본 설정을 합니다.
    with Image.open(png_files[0]) as first_image:
        # 이미지 크기 조정
        width, height = first_image.size
        resized_images = []

        # 각 PNG 파일을 GIF에 추가하기 위해 크기를 조정합니다.
        for png_file in png_files:
            with Image.open(png_file) as image:
                resized_image = image.resize((width, height))
                resized_images.append(resized_image)

        # GIF 파일로 저장합니다.
        resized_images[0].save(
            gif_name,
            format="GIF",
            append_images=resized_images[1:],
            save_all=True,
            duration=frame_duration,
            loop=0,
        )

    print("GIF 파일이 생성되었습니다.")