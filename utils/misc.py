# -*- coding: utf-8 -*-
'''VocalPy - A python version based on VocalMat'''

__email__ = 'gustavo.santana@yale.edu'
__license__ = 'Apache License, Version 2.0'
__copyright__ = '2020 Dietrich Lab - Yale University School of Medicine'

import os
import logging

def create_logger(args=None, out_dir=None):
    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(
                                    '{0}/{1}.log'.format(out_dir, 'output')),
                                logging.StreamHandler()
                            ])
        logging.info('verbose output on')
    else:
        print('logging to file: {}'.format(
            os.path.join(out_dir, 'output.log')))
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)-5.5s]  %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(
                                    '{0}/{1}.log'.format(out_dir, 'output')),
                            ])
