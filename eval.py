import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  # 장기하와 얼굴들 ㅋ 가사:
  '너는 쿨쿨 자나봐.',
  '정성스럽께 저걷떤 거야.',
  '이모티콘 하나마저 조심스럽께 정헤써.',

  '모든 걸 마무리헤버려써.',
  '나는 큰 결시믈 하고서 보넨 문잔데.',
  # 장기하와 얼굴들 새해복 가사:
  '완전히 쾅 다친 데화창뿌니네.',
  '세헤봉 마니 마느세요.',
  '세헤 봉마느로느 난되.',
  '너가 잘헤야지',
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
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
