import os
import sys
from text_mining.src.Text_Mining_Pipeline import text_mining_pipeline

if __name__ == '__main__':
    print('starting')

    Text_Mining_Log_Dir = 'Text_Mining_Logs'

    starter_dir = os.path.abspath(os.path.dirname(__file__))

    args = sys.argv

    if not os.path.isdir(os.path.join(starter_dir, Text_Mining_Log_Dir)):
        os.mkdir(Text_Mining_Log_Dir)

    text_mining_pipeline(args=args, log_dir=os.path.join(starter_dir, Text_Mining_Log_Dir))
