from njuner import NJUNER
import os
import time
import argparse
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_dir", default=None, type=str, required=True, help="The model directory.")
    parser.add_argument("--input_file", default=None, type=str, required=True, help="The input file to predict.")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the result files will be written.")
    # Other parameters
    parser.add_argument("--conll_format", default=False, action='store_true',
                        help="Whether the input file is in conll format.")
    parser.add_argument("--batch_size", default=8, type=int, help="Total batch size for predict.")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available.")
    args = parser.parse_args()
    if not os.path.exists(args.model_dir):
        logging.error("%s is not a valid path." % args.model_dir)
        exit(1)
    if not os.path.exists(args.input_file):
        logging.error("%s is not a valid path." % args.input_file)
        exit(1)
    start_time = time.time()
    njuner = NJUNER(model_dir=args.model_dir, batch_size=args.batch_size, no_cuda=args.no_cuda)
    njuner.predict_file(args.input_file, args.output_dir, args.conll_format)
    logging.info("Elapsed time: %f", time.time()-start_time)
