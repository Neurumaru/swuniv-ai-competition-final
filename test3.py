import os
import re
import cv2
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from jamo import h2j, j2hcj
from join_jamos import join_jamos
from sklearn.model_selection import train_test_split
from trdg.generators import GeneratorFromStrings, GeneratorFromDict
from deep_text_recognition_benchmark.create_lmdb_dataset import createDataset

def data_generate(char, fonts_list, orientation=0):
    words = np.random.choice([1, 2, 3])
    fonts = [np.random.choice(fonts_list)]
    size = int(np.random.normal(300, 50))
    blur = np.random.choice([True, False])
    background_type = np.random.choice([0])
    distorsion_type = np.random.choice([0, 1, 2, 3])
    distorsion_orientation = np.random.choice([0, 1, 2])
    text_color = f'#{np.random.randint(0, 256):02X}{np.random.randint(0, 256):02X}{np.random.randint(0, 256):02X}'
    character_spacing = np.random.randint(0, 100)
    skewing_angle = np.random.randint(0, 10)
    margins = (
        size * np.random.randint(-5, 15) // 100,
        size * np.random.randint(-5, 15) // 100,
        size * np.random.randint(-5, 15) // 100,
        size * np.random.randint(-5, 15) // 100
    )
    stroke_width = np.random.choice([0, 0, 0, 3, 6])
    stroke_fill = f'#{np.random.randint(0, 256):02X}{np.random.randint(0, 256):02X}{np.random.randint(0, 256):02X}'

    if char is not None:
        length = np.random.randint(1, 8 // words)
        strings = ' '.join([''.join(list(np.random.choice(char, length))) for _ in range(words)])
        generator = GeneratorFromStrings(
            [strings],
            fonts=fonts,
            language='ko',
            size=size,
            skewing_angle=skewing_angle,
            random_skew=True,
            blur=blur,
            background_type=background_type,
            distorsion_type=distorsion_type,
            distorsion_orientation=distorsion_orientation,
            text_color=text_color,
            orientation=orientation,
            character_spacing=character_spacing,
            margins=margins,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
    else:
        generator = GeneratorFromDict(
            fonts=fonts,
            language='ko',
            size=size,
            skewing_angle=skewing_angle,
            random_skew=True,
            blur=blur,
            # background_type=background_type,
            # distorsion_type=distorsion_type,
            # distorsion_orientation=distorsion_orientation,
            text_color=text_color,
            orientation=orientation,
            character_spacing=character_spacing,
            margins=margins,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    for img, lbl in generator:
        return np.array(img), lbl

def save_gt(filename, img_path_list, text_list, original=False, negative_img_path_list=None):
    if negative_img_path_list is not None:
        img_path_list = img_path_list + list(negative_img_path_list)
        text_list = text_list + ['.' for _ in range(len(negative_img_path_list))]
    with open(filename, 'w', encoding='utf-8') as f:
        for img_path, text in zip(img_path_list, text_list):
            if original:
                f.write(f'{img_path}\t{text}\n')
            else:
                f.write(f'{img_path}\t{j2hcj(h2j(text))}\n')

inputs = 'inputs'
outputs = 'outputs'
processing = 'processing'
history = 'history'

train_csv = 'train.csv'
test_csv = 'test.csv'
submission_csv = 'sample_submission.csv'

separation = 'separation'
generate_jamo = 'generate_jamo'
generate_freq = 'generate_freq'
generate_dict = 'generate_dict'
lmdb = 'lmdb'
craft = 'CRAFT/data_root_dir'

horizontal = 'horizontal'
vertical = 'vertical'

craft_train = 'ch4_training_images'
craft_test = 'ch4_test_images'
craft_train_gt = 'ch4_training_localization_transcription_gt'
craft_test_gt = 'ch4_test_localization_transcription_gt'

train_gt = 'train_gt.txt'
neg_train_gt = 'neg_train_gt.txt'
test_gt = 'test_gt.txt'
valid_gt = 'valid_gt.txt'
valid_no_space_gt = 'valid_no_space_gt.txt'
training_gt = 'training_gt.txt'
neg_training_gt = 'neg_training_gt.txt'
validation_gt = 'validation_gt.txt'
generate_gt = 'generate_gt.txt'

first = [*'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ']
middle = [*'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ']
last = [''] + [*'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ']

jamo_ko = [join_jamos(f'{f}{m}{l}') for f in first for m in middle for l in last] + [' ']
freq_ko = [*'가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘']
fonts_list = list(map(lambda x : os.path.join('trdg/fonts/ko/', x), os.listdir('trdg/fonts/ko/')))

img_list, lbl_list = [], []
img_path_list, text_list = [], []

shutil.rmtree(os.path.join(processing, generate_jamo, horizontal))
for idx in tqdm(range(1000)):
    img_path = f'./{generate_jamo}/{idx:06d}.png'
    img, lbl = data_generate(jamo_ko, fonts_list, orientation=1)
    while img.ndim == 0:
        img, lbl = data_generate(jamo_ko, fonts_list, orientation=1)
    os.makedirs(os.path.dirname(os.path.join(processing, generate_jamo, horizontal, img_path)), exist_ok = True)
    cv2.imwrite(os.path.join(processing, generate_jamo, horizontal, img_path), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    img_path_list.append(img_path)
    text_list.append(lbl)
    
save_gt(os.path.join(processing, generate_jamo, horizontal, generate_gt), img_path_list, text_list)

img_list, lbl_list = [], []
img_path_list, text_list = [], []

shutil.rmtree(os.path.join(processing, generate_jamo, vertical))
for idx in tqdm(range(1000)):
    img_path = f'./{generate_jamo}/{idx:06d}.png'
    img, lbl = data_generate(jamo_ko, fonts_list, orientation=0)
    while img.ndim == 0:
        img, lbl = data_generate(jamo_ko, fonts_list, orientation=0)
    os.makedirs(os.path.dirname(os.path.join(processing, generate_jamo, vertical, img_path)), exist_ok = True)
    cv2.imwrite(os.path.join(processing, generate_jamo, vertical, img_path), img)
    img_path_list.append(img_path)
    text_list.append(lbl)
    
save_gt(os.path.join(processing, generate_jamo, vertical, generate_gt), img_path_list, text_list)

img_list, lbl_list = [], []
img_path_list, text_list = [], []

shutil.rmtree(os.path.join(processing, generate_freq, horizontal))
for idx in tqdm(range(1000)):
    img_path = f'./{generate_freq}/{idx:06d}.png'
    img, lbl = data_generate(freq_ko, fonts_list, orientation=1)
    while img.ndim == 0:
        img, lbl = data_generate(freq_ko, fonts_list, orientation=1)
    os.makedirs(os.path.dirname(os.path.join(processing, generate_freq, horizontal, img_path)), exist_ok = True)
    cv2.imwrite(os.path.join(processing, generate_freq, horizontal, img_path), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    img_path_list.append(img_path)
    text_list.append(lbl)
    
save_gt(os.path.join(processing, generate_freq, horizontal, generate_gt), img_path_list, text_list)

img_list, lbl_list = [], []
img_path_list, text_list = [], []

shutil.rmtree(os.path.join(processing, generate_freq, vertical))
for idx in tqdm(range(1000)):
    img_path = f'./{generate_freq}/{idx:06d}.png'
    img, lbl = data_generate(freq_ko, fonts_list, orientation=0)
    while img.ndim == 0:
        img, lbl = data_generate(freq_ko, fonts_list, orientation=0)
    os.makedirs(os.path.dirname(os.path.join(processing, generate_freq, vertical, img_path)), exist_ok = True)
    cv2.imwrite(os.path.join(processing, generate_freq, vertical, img_path), img)
    img_path_list.append(img_path)
    text_list.append(lbl)
    
save_gt(os.path.join(processing, generate_freq, vertical, generate_gt), img_path_list, text_list)