import re
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from random import *
import copy

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, args, tokenizer, data, mode):
        self.args = args

        self.tokenizer = tokenizer
        self.max_mask_size = int(args.max_len * 0.15)

        self.input_ids = []
        self.decoder_input_ids = []
        self.labels = []

        self.masked_source = []
        self.masked_tokens = []
        self.masked_position = []

        self.infilling_source = []

        self.mode = mode

        if self.args.mode == 'train':
            self.source_data = data[0]
            # mlm일땐 지우고 아니면 열고
            self.target_data = data[1]
        else:
            self.source_data = data

        """
        init token, idx = [CLS], 0
        pad token, idx = [PAD], 1
        unk token, idx = [UNK], 3
        eos token, idx = [EOS], 2
        sep = 4
        mask = 5
        
        klue
        <s> 0
        <pad>   1
        <eos>   2
        <unk>   3
        [EOS]   50265
        <mask>  50264
        
        """
        # self.special_tokens_dict = {'eos_token': '[EOS]'}
        # self.tokenizer.add_special_tokens(self.special_tokens_dict)

        self.init_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.eos_token = self.tokenizer.eos_token
        self.mask_token = self.tokenizer.mask_token

        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        self.eos_token_idx = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.mask_token_idx = self.tokenizer.convert_tokens_to_ids(self.mask_token)

    def load_data(self, type):
        
        # text infilling
        """
        for src in self.source_data:
            check, sentence = self.data2tensor_mlm(src)
            if check == None:
                continue

            tensored_src_for_mask = copy.deepcopy(sentence)

            source = self.get_text_infilling(tensored_src_for_mask)

            self.infilling_source.append(source)
        """
        # mlm
        """
        for src in self.source_data:
            check, sentence = self.data2tensor_mlm(src)
            if check == None:
                continue

            tensored_src_for_mask = copy.deepcopy(sentence)

            masked_source, masked_tokens, masked_position = \
                self.get_masked_source(tensored_src_for_mask)
            self.masked_source.append(masked_source)
            self.masked_tokens.append(masked_tokens)
            self.masked_position.append(masked_position)
        """

        # summ
        if self.mode != 'test':
            for src, tgt in zip(self.source_data, self.target_data):
                check, sentence = self.data2tensor(src, tgt)
                if check == None:
                    continue
                tensored_src_for_mask = copy.deepcopy(sentence)

                masked_source, masked_tokens, masked_position = \
                    self.get_masked_source(tensored_src_for_mask)
                self.masked_source.append(masked_source)
                self.masked_tokens.append(masked_tokens)
                self.masked_position.append(masked_position)

        else:
            for src in self.source_data:
                check, sentence = self.data2tensor(src, None)

                tensored_src_for_mask = copy.deepcopy(sentence)

                masked_source, masked_tokens, masked_position = \
                    self.get_masked_source(tensored_src_for_mask)
                self.masked_source.append(masked_source)
                self.masked_tokens.append(masked_tokens)
                self.masked_position.append(masked_position)

    def preprocess_text(self, text):
        """
            Preprocessing

            EX) INPUT
            #@이름# 어제 그 동묘 고양이사건전말 떳다 ㅋㅋㅋ헐모양근데 사진은 잇어서 내용 아무렇게 올린덧 헐 한국사 공부하면
             화가 많아지는건가...ㅋㅋㅋㅋㅋㅋㅋㅋ맞아...근데 나중에한국사 배우면친일청산안한거너무 답답하다ㅠ
            박근혜 정부 배울꺼잖아얼마나 븅신 같을까.... 그래도 탄핵됏으니.... 이게 가능하다고.........? ㅇ0ㅇ이 반응일듯 합격예상??어

            EX) OUTPUT
            어제 그 동묘 고양이사건전말 떳다 헐모양근데 사진은 잇어서 내용 아무렇게 올린덧 헐 한국사 공부하면 화가 많아지는건가 맞아 근데 나중에한국사
            배우면친일청산안한거너무 답답하다 박근혜 정부 배울꺼잖아얼마나 븅신 같을까 그래도 탄핵됏으니 이게 가능하다고 0 이 반응일듯 합격예상 어
        """
        text = re.sub("<br />", " ", text)
        text = re.sub("#@[가-힣]+#", " ", text)
        text = re.sub("[ㄱ-ㅎㅏ-ㅣ]+", " ", text)
        text = re.sub("[^A-Za-z0-9가-힣]", " ", text)
        tokens = text.lower().split()

        return (" ".join(tokens))

    def stop_word(self, text):

        stop_words = '아 휴 아이구 아이쿠 아이고 어 헐 헉 또옹 웅 옹 앙 잉 헹 홍 훙 대박 이새낔 새끼 \xa0 시밬 시벌 응 니니해 니니 힝힝 휴 엉 증말 정말 옹옹 ' \
                     '애스야스 미친 시끼 얔 워 워워 응응 존나 짱 짱나 진짜 찐 엿 픽 폭 튀 튀튀 툭 톡 팡 풍 퓨 큐 띵 띠용 용 또옹' \
                     ' 뚱 뷁 쉙 뒑 아녀 호구 놉 눕 omg OMG 야 짱나 존좋 야쓰 야쓰야쓰 증말 ' \
                     '그쵸 그쵸그쵸 마쟈 마쟈마쟈 우웅 진짴 녜 네 넘 엥 왜 url 띠 얍 얍얍 어케 어케요 왜엥 왜왱 아놔' \
                     '주르륵 또르륵 또르르 좋아 쩔어 쩐다 음 아니 와우 좋아좋아 이지랄 오호 왜왜 하 허 허허 시발 쇼발 십알 허허허 흑 흑흑 ' \
                     '띡 딱 똑 이런 아옿 오오 dh 오 오오오 아오 우응 뿌엥 뿌 뽀 빠 젠장 아눂 오잉 또잉 오 하잇 와따시와 곰방와 헣 흡 개꿀 우우 개빡쳐 개뽝친 린정 인정' \
                     '등등 등 근데 크 크크 크크크 흐 흐흐 흐흐흐 나중에 얼마나 많이 그 노노 뎨발요 어케 으궁 또 앜 너무 븅신 병신 개새 개샠 모양 이게 에게 '

        word_tokens = word_tokenize(text)
        stop_words = stop_words.split(' ')
        result = []
        for w in word_tokens:
            if w not in stop_words:
                result.append(w)

        return (" ".join(result))

    def data2tensor(self, src, tgt):

        if self.args.mode == 'train':

            source, target = self.preprocess_text(src.strip()), self.preprocess_text(tgt.strip())

            source = self.stop_word(source)
            target += '.'
            # source = source.replace('rlaqhdals', '[MASK]')

            input_ids = self.tokenizer.encode(source)[1:-1]

            if len(input_ids) <= 2:
                return None, None

            if len(input_ids) >= self.args.max_len:
                input_ids = input_ids[:self.args.max_len]
            else:
                input_ids = self.add_padding_data(input_ids)

            dec_input_ids = self.tokenizer.encode(target)[:-1]

            if len(dec_input_ids) >= self.args.max_len:
                dec_input_ids = dec_input_ids[:self.args.max_len - 1] + [self.eos_token_idx]
            else:
                dec_input_ids = dec_input_ids + [self.eos_token_idx]

            label_ids = dec_input_ids[1:]

            dec_input_ids = self.add_padding_data(dec_input_ids)
            label_ids = self.add_padding_data(label_ids)

            input_ids = torch.LongTensor(input_ids)
            dec_input_ids = torch.LongTensor(dec_input_ids)
            label_ids = torch.LongTensor(label_ids)

            self.input_ids.append(input_ids)
            self.decoder_input_ids.append(dec_input_ids)
            self.labels.append(label_ids)

        else:
            source = self.preprocess_text(src.strip())
            source = self.stop_word(source)

            # source = source.replace('rlaqhdals', '[MASK]')

            input_ids = self.tokenizer.encode(source)[1:-1]

            if len(input_ids) >= self.args.max_len:
                input_ids = input_ids[:self.args.max_len]
            else:
                input_ids = self.add_padding_data(input_ids)

            input_ids = torch.torch.LongTensor(input_ids)
            self.input_ids.append(input_ids)

        return True, input_ids

    def data2tensor_mlm(self, src):
        source = self.preprocess_text(src.strip())

        source = self.stop_word(source)

        input_ids = self.tokenizer.encode(source)[1:-1]

        if len(input_ids) <= 2:
            return None, None

        if len(input_ids) >= self.args.max_len:
            input_ids = input_ids[:self.args.max_len]
        else:
            input_ids = self.add_padding_data(input_ids)

        dec_input_ids = self.tokenizer.encode(source)[:-1]

        if len(dec_input_ids) >= self.args.max_len:
            dec_input_ids = dec_input_ids[:self.args.max_len - 1] + [self.eos_token_idx]
        else:
            dec_input_ids = dec_input_ids + [self.eos_token_idx]

        label_ids = dec_input_ids[1:]

        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_padding_data(label_ids)

        input_ids = torch.LongTensor(input_ids)
        dec_input_ids = torch.LongTensor(dec_input_ids)
        label_ids = torch.LongTensor(label_ids)

        self.input_ids.append(input_ids)
        self.decoder_input_ids.append(dec_input_ids)
        self.labels.append(label_ids)

        return True, torch.LongTensor(input_ids).squeeze(0)

    def get_text_infilling(self, source):
        ori_src = copy.deepcopy(source)

        try:
            start_padding_idx = (source == self.pad_token_idx).nonzero()[0].data.cpu().numpy()[0]
        except IndexError:
            start_padding_idx = self.args.max_len

        source = source[:start_padding_idx]
        max_masking_len = max(1, int(round(len(source) * 0.3)))

        step = 0
        lambda_ = 3
        poisson_list = []
        poisson_zero_checker = 0
        while step < max_masking_len:
            poisson_idx = int(torch.poisson(torch.rand(1) * lambda_))

            if poisson_idx == 0:
                poisson_zero_checker += 1
                if poisson_zero_checker >= int(max_masking_len*0.1):
                    continue

            poisson_list.append(poisson_idx)
            if poisson_idx == 0:
                step += 1
            else:
                step += poisson_idx

            check_poisson_zero = len([val for val in poisson_list if val == 0])
            check_poisson_non_zero = sum([val for val in poisson_list if val != 0])

            if check_poisson_non_zero + check_poisson_zero >= max_masking_len:
                poisson_list = poisson_list[:-1]

        check_poisson_zero = len([val for val in poisson_list if val == 0])
        check_poisson_non_zero = sum([val for val in poisson_list if val != 0])

        assert check_poisson_non_zero >= check_poisson_zero

        mask_index = []
        source_index = torch.arange(len(source)).tolist()

        stop = 0
        for poisson in poisson_list:
            random_index = choice(source_index)

            if poisson == 0:
                poisson_range = [val for val in range(random_index, random_index + 1)]
                poisson_range.append('ZeroPoisson')

            else:
                while 1:
                    stop += 1
                    if stop == 1000:
                        break
                    cur_range = [val for val in range(random_index, random_index + poisson)]

                    if list(set(cur_range) & set(source_index)) == cur_range:
                        poisson_range = [val for val in range(random_index, random_index + poisson)]
                        break

                    else:
                        random_index = choice(source_index)

                if stop == 1000:
                    break

            if stop == 1000:
                mask_index = []
                print("==WHILE==")
                break

            mask_index.append(poisson_range)
            source_index = [val for val in source_index if val not in poisson_range]

        if len(mask_index) == 0:
            mask_index = [[0], [-111]]
        else:
            mask_index = sorted((mask_index)) + [[-111]]

        p_range = [-101]
        text_infilling_list = []
        step_checker = 1
        for step, value in enumerate(ori_src):

            if len(mask_index) == 1 and step in p_range:
                continue
            elif len(mask_index) == 1 and step not in p_range:
                text_infilling_list.append(value)
                continue

            if step == mask_index[0][0] and mask_index[0][-1] != 'ZeroPoisson' and step not in p_range:
                p_range = mask_index[0]
                text_infilling_list.append(self.mask_token_idx)
                mask_index = mask_index[1:]

            elif step == mask_index[0][0] and mask_index[0][-1] == 'ZeroPoisson' and step not in p_range:
                p_range = mask_index[0]
                text_infilling_list.append(self.mask_token_idx)
                text_infilling_list.append(value)
                mask_index = mask_index[1:]

            else:
                if step in p_range:
                    step_checker += 1

                    if step_checker == len(p_range):
                        mask_index = mask_index[1:]
                        step_checker = 0
                    continue

                text_infilling_list.append(value)

        tensored = torch.LongTensor([text_infilling_list]).squeeze(0)

        try:
            assert len(tensored) <= self.args.max_len
        except AssertionError:
            print(ori_src)
            print(tensored)
            print(len(ori_src))
            print(len(tensored))
            exit()

        tensored = torch.LongTensor(self.add_padding_data(tensored))

        return tensored

    def get_masked_source(self, source):

        ori_src = copy.deepcopy(source)

        try:
            start_padding_idx = (source == self.pad_token_idx).nonzero()[0].data.cpu().numpy()[0]
        except IndexError:
            start_padding_idx = self.args.max_len

        source = source[:start_padding_idx]

        n_pred = min(self.max_mask_size, max(1, int(round(len(source) * 0.15))))  # mask 15%
        cand_maked_pos = [i for i, token in enumerate(source)]

        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []

        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(ori_src[pos])
            if random() < 0.8:  # 80%
                source[pos] = self.mask_token_idx
            elif random() < 0.5:  # 10%
                index = randint(0, len(self.tokenizer) - 1)
                source[pos] = index

        masked_source = list(copy.deepcopy(source.data.numpy()))
        for i in range(self.args.max_len - len(source)): masked_source.append(self.pad_token_idx)

        # Zero Padding (100% - 15%) tokens
        if self.max_mask_size > n_pred:
            n_pad = self.max_mask_size - n_pred
            masked_tokens.extend([self.pad_token_idx] * n_pad)
            masked_pos.extend([self.pad_token_idx] * n_pad)

        return torch.LongTensor(masked_source), torch.LongTensor(masked_tokens), torch.LongTensor(masked_pos)

    def add_padding_data(self, inputs):
        if len(inputs) <= self.args.max_len:
            pad = np.array([self.pad_token_idx] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            print("?????????")
            exit()
            # inputs = inputs[:self.args.max_len-1] + [self.eos_token_idx]

        return inputs

    def __getitem__(self, index):

        if self.args.mode == 'train':

            input_data = {'input_ids': self.input_ids[index].to(self.args.device),
                          'decoder_input_ids': self.decoder_input_ids[index].to(self.args.device),
                          'labels': self.labels[index].to(self.args.device)}
        else:
            input_data = {'input_ids': self.input_ids[index].to(self.args.device)}

        auxiliary_data = {
            'source': self.masked_source[index].to(self.args.device),
            'tokens': self.masked_tokens[index].to(self.args.device),
            'position': self.masked_position[index].to(self.args.device)}
        """
        auxiliary_data = {
            'source': self.infilling_source[index].to(self.args.device)}
        """
        return input_data, auxiliary_data

    def __len__(self):
        return len(self.input_ids)


def get_loader(args, tokenizer, data, mode='train_val'):

    data_iter = ModelDataLoader(args, tokenizer, data, mode)
    data_iter.load_data(mode)

    if args.mode == 'train':
        loader = DataLoader(dataset=data_iter,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)
    else:
        loader = DataLoader(dataset=data_iter,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0)
    return loader, tokenizer


if __name__ == '__main__':
    get_loader('test')