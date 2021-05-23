import os

class Logger:
    def __init__(self, filename):
        self.filename = filename
        if not os.path.isfile(filename):
            with open(file=filename, mode='w') as f:
                f.close()

    def write_line(self, log):
        with open(file=self.filename, mode='a') as f:
            f.writelines(log + '\n')

    def write_epoch_loss(self, epoch, loss):
        with open(file=self.filename, mode='a') as f:
            f.writelines(str(epoch) + ' ' + str(round(loss, 6)) + '\n')

def main():
    log = Logger('log.txt')
    log.write_line('1')
    log.write_line('2')
    log.write_line('3')

if __name__ == '__main__':
    main()