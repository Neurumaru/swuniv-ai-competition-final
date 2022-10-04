import os
import cv2
import xml.etree.ElementTree as ET
from join_jamos import join_jamos
from tqdm import tqdm

DATA = 'inputs/KAIST'
OUT_DIR = 'processing/KAIST'

images = 'images'
gt = 'localization_transcription_gt'

first = [*'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ']
middle = [*'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ']
last = [''] + [*'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ']

jamo_ko = [join_jamos(f'{f}{m}{l}') for f in first for m in middle for l in last]

def main():
    os.makedirs(os.path.join(OUT_DIR, images), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, gt), exist_ok=True)

    for filename in tqdm(os.listdir(DATA)):
        text, ext = os.path.splitext(filename)

        if ext == '.xml':
            try:
                with open(os.path.join(DATA, filename), 'r', encoding='utf-8') as xml:
                    lines = xml.readlines()
                root = ET.fromstringlist(lines)
            except:
                continue
            image = root.find('image')
            resolution = image.find('resolution')
            resolution_x, resolution_y = int(resolution.get('x')), int(resolution.get('y'))

            img = cv2.imread(os.path.join(DATA, f'{text}.jpg'))
            img_y, img_x, _ = img.shape
            
            if img_x == resolution_x and img_y == resolution_y:
                cv2.imwrite(os.path.join(OUT_DIR, images, f'{text}.jpg'), img)
            elif img_x == resolution_y and img_y == resolution_x:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(os.path.join(OUT_DIR, images, f'{text}.jpg'), img)
            else:
                continue

            words = image.find('words')
            with open(os.path.join(OUT_DIR, gt, f'gt_{text}.txt'), 'w') as f:
                for word in words:
                    for character in word.findall('character'):
                        x = int(character.get('x'))
                        y = int(character.get('y'))
                        width = int(character.get('width'))
                        height = int(character.get('height'))
                        char = character.get('char')
                        
                        if char in jamo_ko:
                            f.write(f'{x},{y},{x+width},{y},{x+width},{y+height},{x},{y+height},{char}\n')

if __name__ == '__main__':
    main()