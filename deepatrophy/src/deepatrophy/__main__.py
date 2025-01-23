import argparse
from deepatrophy.neck_trim import NeckTrimLauncher  
from deepatrophy.global_alignment import GlobalAlignmentLauncher
from deepatrophy.deep_atrophy_train import DeepAtrophyTrainLauncher
from deepatrophy.deep_atrophy_test import DeepAtrophyTestLauncher
from deepatrophy.PAIIR import PAIIRLauncher

# Create a parser
parse = argparse.ArgumentParser(
    prog="deepatrophy", description="DEEPATROPHY: a Longitudinal Package for Brain Progression Estimation")

# Add subparsers for the main commands
sub = parse.add_subparsers(dest='command', help='sub-command help', required=True)

# Add the DeepAtrophy subparser commands
c_neck_trim = NeckTrimLauncher(
    sub.add_parser('neck_trim', help='For each scan, perform neck trimming'))

c_global_align = GlobalAlignmentLauncher(
    sub.add_parser('obtain_image_pair', help='For each scan pair, perform global alignment'))

c_model_train = DeepAtrophyTrainLauncher(
    sub.add_parser('run_training', help='DeepAtrophy model training'))

c_model_test = DeepAtrophyTestLauncher(
    sub.add_parser('run_test', help='DeepAtrophy model testing for the whole dataset'))
    
c_PAIIR = PAIIRLauncher(
    sub.add_parser('PAIIR', help='Predicted-to-Actual Interscan Interval Ratio'))

# Parse the arguments
args = parse.parse_args()
args.func(args)