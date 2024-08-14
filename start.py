import subprocess

def run_train():
    train_args = ["-c", "config/aerial/unetformer.py"]
    subprocess.run(['python', 'train_supervision.py'] + train_args,check=True)

def run_test():
    train_args = ['-c','config/aerial/unetformer.py', '-o','fig_results/aerial/unetformer.py']
    subprocess.run(['python', 'aerial_test.py'] + train_args,check=True)

if __name__ == '__main__':
    run_train()
    run_test()