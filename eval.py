import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


# sentences = [
#     # '완전히 쾅 닫힌 대화창 뿐이네',
#     # '정성스럽게 적었던 거야',
#     # '나는 큰 결심을 하고서 보낸 문잔데',
#     # '모든걸 마무리 해버렸어',
#     # '이모티콘 하나마저 조심스럽게 보냈어',
#     # '너가 잘해야지',
#     # '새해 복만으로는 안돼',
#   # 장기하와 얼굴들 ㅋ 가사:
#   '신진 샹숑가수의 신춘 샹숑쇼우',
#   '철수 책상 철 책상',
#   '창경원 창살은 쌍창살',
#   '스위스에서 온 스미스씨',
#   # 장기하와 얼굴들 새해복 가사:
#   '간장 공장 공장장',
#   '한양양장점 옆 한양양장점',
#   '후회한 시간을 후회할 거잖아',
# ]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-char-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    sentences=[]
    with open('./eval_char.txt', encoding='utf-8') as f:
        for line in f:
            try:
                parts = line.strip().replace('"', '').split('|')
                text = parts[3]
                sentences.append(text)
            except:
                pass
    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
            f.write(synth.synthesize(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
  main()
