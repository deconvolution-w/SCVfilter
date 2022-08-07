<<<<<<< HEAD
from torch.utils.data import Dataset, DataLoader
import torch

class myDataset(Dataset):
    def __init__(self, trainOrtest, nclass, split_rate = 0.7):
        assert trainOrtest in ('train', 'test'), 'The parameter must be train or test'
        self.train = trainOrtest  # train = 1  test = 0
        self.train = 1 if trainOrtest == 'train' else 0
        self.splitRate = split_rate
        self.labels = []
        self.X = []
        self.Y = []
        #########################################################################
        if nclass == 3:
            main_land = open('./data/3-class/mainland_sequences.fasta', 'r+')
            taiwan_land = open('./data/3-class/taiwan_sequences.fasta', 'r+')
            hongkong_land = open('./data/3-class/hongkong_sequences.fasta', 'r+')
            self.get_data(main_land, 0, self.train) #1792
            self.get_data(taiwan_land, 1, self.train)
            self.get_data(hongkong_land, 2, self.train)  #340
        #########################################################################
        if nclass == 6:
            africa = open('./data/6-class/africa.sequences.fasta', 'r+')
            self.get_data(africa, 0, self.train)
            asia = open('./data/6-class/asia.sequences.fasta', 'r+')
            self.get_data(asia, 1, self.train)
            europe = open('./data/6-class/europe.sequences.fasta', 'r+')
            self.get_data(europe, 2, self.train)
            northamerica = open('./data/6-class/northamerica.sequences.fasta', 'r+')
            self.get_data(northamerica, 3, self.train)
            oceania = open('./data/6-class/oceania.sequences.fasta', 'r+')
            self.get_data(oceania, 4, self.train)
            sorthamerica = open('./data/6-class/sorthamerica.sequences.fasta', 'r+')
            self.get_data(sorthamerica, 5, self.train)
        #########################################################################
        if nclass == 8:
            alpha = open('./data/8-class/vocalphagry.sequences.fasta', 'r+')
            self.get_data(alpha, 0, self.train)
            beta = open('./data/8-class/vocbetagh.sequences.fasta', 'r+')
            self.get_data(beta, 1, self.train)
            delte = open('./data/8-class/vocdeltegk.sequences.fasta', 'r+')
            self.get_data(delte, 2, self.train)
            gamma = open('./data/8-class/vocgammagr.sequences.fasta', 'r+')
            self.get_data(gamma, 3, self.train)
            omicron = open('./data/8-class/vocomicronGRA.sequences.fasta', 'r+')
            self.get_data(omicron, 4, self.train)
            lambda_ = open('./data/8-class/voilambdagr.sequences.fasta', 'r+')
            self.get_data(lambda_, 5, self.train)
            mu = open('./data/8-class/voimugh.sequences.fasta', 'r+')
            self.get_data(mu, 6, self.train)
            gh = open('./data/8-class/vumgh490R.sequences.fasta', 'r+')
            self.get_data(gh, 7, self.train)
        #########################################################################
        self.X, self.Y = torch.Tensor(self.X), torch.Tensor(self.Y)
        print('The size of {}data is [{},{}]'.format(trainOrtest, self.X.size(0),self.X.size(1)))



    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        return X.long(), Y.long()

    def __len__(self):
        return len(self.X)

    def str2int(self,str):
        atcg2int = {'N': 1, 'A': 2, 'T': 3, 'C': 4,'G': 5}
        int_seq = [atcg2int[i] for i in str]
        return int_seq

    def length(self,int_seq):
        if len(int_seq) <= 30100:
            int_seq = int_seq + [0] * (30100 - len(int_seq))
        else:
            int_seq = int_seq[:30100]
        return int_seq

    def data_process(self, str):
        str = self.seq_init(str)
        data = self.str2int(str)
        data = self.length(data)
        return data

    def get_data(self, data, label, train):
        i = 0
        tmp = self.splitRate * 10
        for line in data:
            if i % 20 < 2 * tmp and train:
                if line.startswith('>'):
                    self.Y.append(label)
                else:
                    self.X.append(self.data_process(line.strip()))
            if i % 20 >= 2 * tmp and train == 0:
                if line.startswith('>'):
                    self.Y.append(label)
                else:
                    self.X.append(self.data_process(line.strip()))
            i += 1
    def seq_init(self, seq):
        seq = seq.replace("R", "N")
        seq = seq.replace("C", "N")
        seq = seq.replace("S", "N")
        seq = seq.replace("W", "N")
        seq = seq.replace("Y", "N")
        seq = seq.replace("M", "N")
        seq = seq.replace("K", "N")
        seq = seq.replace("H", "N")
        seq = seq.replace("B", "N")
        seq = seq.replace("D", "N")
        seq = seq.replace("V", "N")
        return seq



if __name__ == '__main__':
    train_data = myDataset('train', 3)
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    train_data = myDataset('test', 3)
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    for x, y in trainloader:
        # print(x, y)
